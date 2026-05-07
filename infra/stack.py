"""CDK stack for the Splendor game webapp deployment."""

from __future__ import annotations

import pathlib

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_integrations as apigwv2_int,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

from infra.models_config import DEPLOYED_CHECKPOINTS

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class SplendorStack(Stack):
    """AWS infrastructure for the Splendor game webapp."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- DynamoDB Tables ---

        games_table = dynamodb.Table(
            self,
            "GamesTable",
            partition_key=dynamodb.Attribute(
                name="game_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )
        games_table.add_global_secondary_index(
            index_name="user_sub-updated_at-index",
            partition_key=dynamodb.Attribute(
                name="user_sub", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="updated_at", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        users_table = dynamodb.Table(
            self,
            "UsersTable",
            partition_key=dynamodb.Attribute(
                name="username", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- S3 Buckets ---

        frontend_bucket = s3.Bucket(
            self,
            "FrontendBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        models_bucket = s3.Bucket(
            self,
            "ModelsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        # --- Lambda Function (Docker image for PyTorch support) ---

        api_function = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                str(_PROJECT_ROOT),
                file="infra/lambda.Dockerfile",
                exclude=[
                    "agent/runs",
                    "replay_webapp/node_modules",
                    ".venv",
                    ".git",
                    ".kiro",
                    ".pytest_cache",
                    "cdk.out",
                    "*.egg-info",
                    "infra/layer",
                ],
            ),
            memory_size=2048,
            timeout=Duration.seconds(60),
            environment={
                "GAMES_TABLE": games_table.table_name,
                "USERS_TABLE": users_table.table_name,
                "MODELS_BUCKET": models_bucket.bucket_name,
                "MODELS_MANIFEST": "checkpoints/manifest.json",
            },
        )

        # Grant Lambda access to DynamoDB tables and models bucket
        games_table.grant_read_write_data(api_function)
        users_table.grant_read_write_data(api_function)
        models_bucket.grant_read(api_function)

        # --- API Gateway HTTP API ---

        http_api = apigwv2.HttpApi(
            self,
            "HttpApi",
            api_name="SplendorApi",
        )

        lambda_integration = apigwv2_int.HttpLambdaIntegration(
            "LambdaIntegration", handler=api_function
        )

        http_api.add_routes(
            path="/api/{proxy+}",
            methods=[apigwv2.HttpMethod.ANY],
            integration=lambda_integration,
        )

        # --- CloudFront Distribution ---

        # API Gateway origin (strip /api prefix is handled by the Lambda router)
        api_origin = origins.HttpOrigin(
            f"{http_api.http_api_id}.execute-api.{self.region}.amazonaws.com",
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        )

        s3_origin = origins.S3BucketOrigin.with_origin_access_control(
            frontend_bucket
        )

        distribution = cloudfront.Distribution(
            self,
            "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=s3_origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
            default_root_object="index.html",
        )

        # --- Frontend Build and Deployment (Task 8.3) ---

        webapp_dist_path = str(_PROJECT_ROOT / "replay_webapp" / "dist")

        s3deploy.BucketDeployment(
            self,
            "FrontendDeployment",
            sources=[
                s3deploy.Source.asset(webapp_dist_path),
            ],
            destination_bucket=frontend_bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        # --- Model Checkpoint Upload (Task 8.4) ---

        import json

        manifest_entries = []
        for ckpt in DEPLOYED_CHECKPOINTS:
            s3_key = f"checkpoints/{ckpt['run']}/{pathlib.Path(ckpt['relative_path']).name}"
            manifest_entries.append(
                {
                    "run": ckpt["run"],
                    "tag": ckpt["tag"],
                    "idx": ckpt["idx"],
                    "s3_key": s3_key,
                    "rating": ckpt.get("rating", 2500.0),
                    "s3_bucket": models_bucket.bucket_name,
                }
            )

        # Upload checkpoint files to models bucket
        checkpoint_sources = []
        for ckpt in DEPLOYED_CHECKPOINTS:
            ckpt_path = _PROJECT_ROOT / ckpt["relative_path"]
            s3_key_prefix = f"checkpoints/{ckpt['run']}"
            checkpoint_sources.append(
                (str(ckpt_path.parent), ckpt_path.name, s3_key_prefix)
            )

        # Deploy each checkpoint to its S3 prefix
        for i, (source_dir, filename, prefix) in enumerate(checkpoint_sources):
            s3deploy.BucketDeployment(
                self,
                f"CheckpointDeployment{i}",
                sources=[s3deploy.Source.asset(source_dir, exclude=["*", f"!{filename}"])],
                destination_bucket=models_bucket,
                destination_key_prefix=prefix,
            )

        # Generate and upload the manifest JSON
        manifest_json = json.dumps(manifest_entries, indent=2)
        s3deploy.BucketDeployment(
            self,
            "ManifestDeployment",
            sources=[s3deploy.Source.data("checkpoints/manifest.json", manifest_json)],
            destination_bucket=models_bucket,
        )

        # --- Outputs ---

        CfnOutput(
            self,
            "ApiEndpointUrl",
            value=http_api.url or "",
            description="API Gateway endpoint URL",
        )

        CfnOutput(
            self,
            "CloudFrontUrl",
            value=f"https://{distribution.distribution_domain_name}",
            description="CloudFront distribution URL",
        )
