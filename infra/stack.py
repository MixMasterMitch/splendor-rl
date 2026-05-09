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
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
)
from constructs import Construct

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

        # --- Lambda Function (Docker image for PyTorch support) ---
        # League checkpoints and data are baked into the container image
        # (no S3 models bucket needed).

        api_function = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                str(_PROJECT_ROOT),
                file="infra/lambda.Dockerfile",
                exclude=[
                    "agent/runs/attn256_v1",
                    "agent/runs/real30_v10",
                    "agent/runs/real30_v11",
                    "agent/runs/real30_v12",
                    "agent/runs/optuna_gpu-tune-cold.db",
                    "agent/runs/league_eval_full.log",
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
            },
        )

        # Grant Lambda access to DynamoDB tables
        games_table.grant_read_write_data(api_function)
        users_table.grant_read_write_data(api_function)

        # Grant Lambda access to Bedrock for LLM-based opponents
        api_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=["*"],
            )
        )

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

        # --- Frontend Build and Deployment ---

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
