from typing import Any

class MessagingServerI:
    def publish(self, channel: str, data: Any) -> None:
        raise NotImplementedError()
        
    def publish_to_group(self, group_id: int, data: Any) -> None:
        raise NotImplementedError()