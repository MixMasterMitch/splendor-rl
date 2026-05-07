"""AWS Bedrock API client with retry logic and observability."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError, ReadTimeoutError, ConnectTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class BedrockResponse:
    """Response from a Bedrock invocation with metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    retries: int
    stop_reason: str = "end_turn"  # "end_turn", "max_tokens", etc.


class BedrockUnavailableError(Exception):
    """Raised when Bedrock is unreachable after exhausting retries."""

    pass


class BedrockClient:
    """AWS Bedrock API client with retry and observability."""

    def __init__(
        self,
        region: str = "us-west-2",
        max_retries_throttle: int = 3,
        max_retries_server_error: int = 1,
    ) -> None:
        self._region = region
        self._max_retries_throttle = max_retries_throttle
        self._max_retries_server_error = max_retries_server_error
        from botocore.config import Config
        config = Config(
            read_timeout=15,
            connect_timeout=5,
            retries={"max_attempts": 0},  # We handle retries ourselves
        )
        self._client = boto3.client("bedrock-runtime", region_name=region, config=config)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> BedrockResponse:
        """Send prompt to Bedrock and return response with metadata.

        Retry behavior:
        - HTTP 429 (throttling): exponential backoff at 1s, 2s, 4s up to max_retries_throttle
        - HTTP 5xx (server error): single retry after 2s up to max_retries_server_error
        - On exhausted retries: raises BedrockUnavailableError
        """
        messages = [{"role": "user", "content": [{"text": user_prompt}]}]
        system = [{"text": system_prompt}]
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        retries = 0
        throttle_retries = 0
        server_error_retries = 0
        start_time = time.perf_counter()

        while True:
            try:
                response = self._client.converse(
                    modelId=model_id,
                    messages=messages,
                    system=system,
                    inferenceConfig=inference_config,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Extract response text
                output_message = response["output"]["message"]
                text = output_message["content"][0]["text"]

                # Extract token usage
                usage = response["usage"]
                input_tokens = usage["inputTokens"]
                output_tokens = usage["outputTokens"]

                # Extract stop reason
                stop_reason = response.get("stopReason", "end_turn")

                logger.info(
                    "Bedrock invocation: model=%s input_tokens=%d output_tokens=%d latency_ms=%.1f retries=%d stop_reason=%s",
                    model_id,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                    retries,
                    stop_reason,
                )

                return BedrockResponse(
                    text=text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    model_id=model_id,
                    retries=retries,
                    stop_reason=stop_reason,
                )

            except ClientError as e:
                error_code = e.response["ResponseMetadata"]["HTTPStatusCode"]

                if error_code == 429:
                    # Throttling - exponential backoff
                    if throttle_retries >= self._max_retries_throttle:
                        raise BedrockUnavailableError(
                            f"Bedrock throttling limit exceeded after {throttle_retries} retries"
                        ) from e
                    delay = 2**throttle_retries  # 1s, 2s, 4s
                    logger.warning(
                        "Bedrock throttled (429), retrying in %ds (attempt %d/%d)",
                        delay,
                        throttle_retries + 1,
                        self._max_retries_throttle,
                    )
                    time.sleep(delay)
                    throttle_retries += 1
                    retries += 1

                elif 500 <= error_code < 600:
                    # Server error - single retry after 2s
                    if server_error_retries >= self._max_retries_server_error:
                        raise BedrockUnavailableError(
                            f"Bedrock server error ({error_code}) after {server_error_retries} retries"
                        ) from e
                    logger.warning(
                        "Bedrock server error (%d), retrying in 2s (attempt %d/%d)",
                        error_code,
                        server_error_retries + 1,
                        self._max_retries_server_error,
                    )
                    time.sleep(2)
                    server_error_retries += 1
                    retries += 1

                else:
                    # Other client errors - don't retry
                    raise BedrockUnavailableError(
                        f"Bedrock client error ({error_code}): {e}"
                    ) from e

            except (ReadTimeoutError, ConnectTimeoutError) as e:
                raise BedrockUnavailableError(
                    f"Bedrock connection/read timeout: {e}"
                ) from e
