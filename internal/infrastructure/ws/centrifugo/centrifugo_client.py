import json
import requests
from typing import Any

class CentrifugoClient:
    def __init__(self, api_url: str, api_key: str, timeout: float = 5.0):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        })

    def publish(self, channel: str, data: Any) -> None:
        """
        Publish data to a channel
        :param channel: Target channel name
        :param data: Data to publish (will be serialized to JSON)
        :raises CentrifugoError: On API error response
        """
        payload = {
            "channel": channel,
            "data": data
        }

        try:
            response = self.session.post(
                f"{self.api_url}/api/publish",
                json=payload,
                timeout=self.timeout
            )
        except requests.exceptions.RequestException as e:
            raise CentrifugoError(f"Network error: {str(e)}") from e

        self._handle_response(response)

    def publish_to_group(self, group_id: int, data: Any) -> None:
        """
        Publish data to a group channel
        :param group_id: Target group ID
        :param data: Data to publish
        """
        channel = f"room:{group_id}"
        self.publish(channel, data)

    def _handle_response(self, response: requests.Response) -> None:
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            raise CentrifugoError(f"Invalid JSON response: {str(e)}") from e

        if response.status_code != 200:
            raise CentrifugoError(
                f"HTTP error {response.status_code}: {response.text}"
            )

        if "error" in response_data and response_data["error"]:
            error = response_data["error"]
            raise CentrifugoError(
                f"Centrifugo error {error['code']}: {error['message']}"
            )

        print("Successfully published to channel")

class CentrifugoError(Exception):
    """Base exception for Centrifugo client errors"""
