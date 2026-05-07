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


def _load_league_checkpoints() -> list[dict]:
    """Read league.json and return entries whose checkpoint files exist on disk."""
    import json

    league_json = _PROJECT_ROOT / "agent" / "runs" / "league" / "league.json"
    if not league_json.exists():
        return []

    with open(league_json) as f:
        data = json.load(f)

    league_dir = league_json.parent
    entries = []
    for entry in data.get("entries", []):
        ckpt_path = league_dir / entry["path"]
        if ckpt_path.exists():
            entries.append({
                "idx": entry["idx"],
                "tag": entry.get("tag", f"idx{entry['idx']}"),
                "path": str(ckpt_path),
                "filename": entry["path"],
                "rating": float(entry.get("rating", 2500.0)),
                "hidden": entry.get("hidden"),
                "arch": entry.get("arch"),
            })
    return entries


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
                "LEAGUE_DATA_KEY": "league/league.json",
            },
        )

        # Grant Lambda access to DynamoDB tables and models bucket
        games_table.grant_read_write_data(api_function)
        users_table.grant_read_write_data(api_function)
        models_bucket.grant_read(api_function)

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

        # --- Model Checkpoint Upload ---
        # Reads league.json at synth time and uploads all checkpoints that
        # exist on disk (the league maintains a rolling window of ~24).

        import json

        league_checkpoints = _load_league_checkpoints()

        manifest_entries = []
        for ckpt in league_checkpoints:
            s3_key = f"checkpoints/league/{ckpt['filename']}"
            manifest_entries.append({
                "run": "league",
                "tag": ckpt["tag"],
                "idx": ckpt["idx"],
                "s3_key": s3_key,
                "rating": ckpt["rating"],
                "hidden": ckpt.get("hidden"),
                "arch": ckpt.get("arch"),
                "s3_bucket": models_bucket.bucket_name,
            })

        # Deploy each checkpoint file to S3
        for i, ckpt in enumerate(league_checkpoints):
            ckpt_dir = str(pathlib.Path(ckpt["path"]).parent)
            filename = ckpt["filename"]
            s3deploy.BucketDeployment(
                self,
                f"CheckpointDeployment{i}",
                sources=[s3deploy.Source.asset(ckpt_dir, exclude=["*", f"!{filename}"])],
                destination_bucket=models_bucket,
                destination_key_prefix="checkpoints/league",
                prune=False,
            )

        # Generate and upload the manifest JSON
        manifest_json = json.dumps(manifest_entries, indent=2)
        s3deploy.BucketDeployment(
            self,
            "ManifestDeployment",
            sources=[s3deploy.Source.data("checkpoints/manifest.json", manifest_json)],
            destination_bucket=models_bucket,
            prune=False,
        )

        # --- League Data Upload ---
        # Upload league.json so the Lambda can compute unified ratings
        # matching the local server's combined_ratings() logic.
        league_json_path = _PROJECT_ROOT / "agent" / "runs" / "league" / "league.json"
        if league_json_path.exists():
            s3deploy.BucketDeployment(
                self,
                "LeagueDataDeployment",
                sources=[s3deploy.Source.asset(
                    str(league_json_path.parent),
                    exclude=["*", "!league.json"],
                )],
                destination_bucket=models_bucket,
                destination_key_prefix="league",
                prune=False,
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
