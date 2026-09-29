"""Notion integration via API.

Supports:
- Read/write pages and databases
- Search across workspace
- Create and update blocks
- Manage databases (query, create entries)

Usage:
    notion = NotionIntegration(account_manager)
    
    # Search
    results = await notion.search("meeting notes", account="bot")
    
    # Read a page
    content = await notion.read_page("page_id", account="bot")
    
    # Write to a page
    await notion.append_blocks("page_id", [
        {"type": "paragraph", "text": "New content here"},
    ], account="bot")
    
    # Query a database
    entries = await notion.query_database("db_id", account="bot", filter={...})
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..core.logging_setup import get_logger

__all__ = ["NotionIntegration", "NotionPage", "NotionDatabase"]

_log = get_logger(__name__)

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


@dataclass
class NotionPage:
    page_id: str
    title: str
    url: str = ""
    created_time: str = ""
    last_edited_time: str = ""
    parent_type: str = ""
    parent_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id, "title": self.title, "url": self.url,
            "created_time": self.created_time, "last_edited_time": self.last_edited_time,
        }


@dataclass
class NotionDatabase:
    database_id: str
    title: str
    url: str = ""
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {"database_id": self.database_id, "title": self.title, "url": self.url}


class NotionIntegration:
    """Notion API integration."""
    
    def __init__(self, account_manager: AccountManager) -> None:
        self.account_manager = account_manager
        _log.info("Notion integration initialized")
    
    async def _api_request(
        self, method: str, endpoint: str, account: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make authenticated Notion API request."""
        cred = self.account_manager.get_credential("notion", account)
        api_key = cred.password
        
        url = f"{NOTION_API}/{endpoint}"
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(
            url, data=body, method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Notion-Version": NOTION_VERSION,
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            _log.error(f"Notion API error: {e.code} - {error_body}")
            raise
    
    # ── Search ───────────────────────────────────────────────────────────────
    
    async def search(
        self, query: str, account: str, *,
        filter_type: str | None = None, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search Notion workspace."""
        data: dict[str, Any] = {"query": query, "page_size": limit}
        
        if filter_type:
            data["filter"] = {"value": filter_type, "property": "object"}
        
        result = await self._api_request("POST", "search", account, data=data)
        
        items = []
        for item in result.get("results", []):
            obj_type = item.get("object")
            if obj_type == "page":
                items.append(self._parse_page(item))
            elif obj_type == "database":
                items.append(self._parse_database(item))
        
        return items
    
    # ── Pages ────────────────────────────────────────────────────────────────
    
    async def read_page(self, page_id: str, account: str) -> NotionPage:
        """Read a page's metadata."""
        result = await self._api_request("GET", f"pages/{page_id}", account)
        return self._parse_page(result)
    
    async def read_page_content(self, page_id: str, account: str) -> list[dict[str, Any]]:
        """Read page content (blocks)."""
        result = await self._api_request("GET", f"blocks/{page_id}/children", account)
        return result.get("results", [])
    
    async def create_page(
        self, parent_id: str, account: str, *,
        title: str = "", properties: dict[str, Any] | None = None,
        content: list[dict[str, Any]] | None = None,
        parent_type: str = "page_id",
    ) -> NotionPage:
        """Create a new page."""
        data: dict[str, Any] = {
            "parent": {parent_type: parent_id},
            "properties": properties or {},
        }
        
        if title and not properties:
            data["properties"] = {
                "title": {"title": [{"text": {"content": title}}]}
            }
        
        if content:
            data["children"] = content
        
        result = await self._api_request("POST", "pages", account, data=data)
        return self._parse_page(result)
    
    async def append_blocks(
        self, page_id: str, blocks: list[dict[str, Any]], account: str,
    ) -> bool:
        """Append blocks to a page."""
        # Convert simple text blocks to Notion format
        notion_blocks = []
        for block in blocks:
            if isinstance(block, str):
                notion_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": block}}]
                    },
                })
            elif isinstance(block, dict):
                if "type" in block and "text" in block:
                    notion_blocks.append({
                        "object": "block",
                        "type": block["type"],
                        block["type"]: {
                            "rich_text": [{"type": "text", "text": {"content": block["text"]}}]
                        },
                    })
                else:
                    notion_blocks.append(block)
        
        data = {"children": notion_blocks}
        await self._api_request("PATCH", f"blocks/{page_id}/children", account, data=data)
        return True
    
    async def update_page_properties(
        self, page_id: str, properties: dict[str, Any], account: str,
    ) -> bool:
        """Update page properties."""
        data = {"properties": properties}
        await self._api_request("PATCH", f"pages/{page_id}", account, data=data)
        return True
    
    # ── Databases ────────────────────────────────────────────────────────────
    
    async def get_database(self, database_id: str, account: str) -> NotionDatabase:
        """Get database metadata."""
        result = await self._api_request("GET", f"databases/{database_id}", account)
        return self._parse_database(result)
    
    async def query_database(
        self, database_id: str, account: str, *,
        filter_obj: dict[str, Any] | None = None,
        sorts: list[dict[str, Any]] | None = None,
        limit: int = 100,
    ) -> list[NotionPage]:
        """Query a database."""
        data: dict[str, Any] = {"page_size": limit}
        
        if filter_obj:
            data["filter"] = filter_obj
        if sorts:
            data["sorts"] = sorts
        
        result = await self._api_request("POST", f"databases/{database_id}/query", account, data=data)
        
        return [self._parse_page(item) for item in result.get("results", [])]
    
    async def create_database_entry(
        self, database_id: str, properties: dict[str, Any], account: str,
    ) -> NotionPage:
        """Create a new entry in a database."""
        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }
        
        result = await self._api_request("POST", "pages", account, data=data)
        return self._parse_page(result)
    
    # ── Parsers ──────────────────────────────────────────────────────────────
    
    def _parse_page(self, item: dict[str, Any]) -> NotionPage:
        # Extract title
        title = ""
        props = item.get("properties", {})
        for key, val in props.items():
            if val.get("type") == "title":
                title_arr = val.get("title", [])
                if title_arr:
                    title = title_arr[0].get("plain_text", "")
                break
        
        parent = item.get("parent", {})
        
        return NotionPage(
            page_id=item.get("id", ""),
            title=title,
            url=item.get("url", ""),
            created_time=item.get("created_time", ""),
            last_edited_time=item.get("last_edited_time", ""),
            parent_type=parent.get("type", ""),
            parent_id=parent.get(parent.get("type", ""), ""),
            properties=props,
        )
    
    def _parse_database(self, item: dict[str, Any]) -> NotionDatabase:
        title_arr = item.get("title", [])
        title = title_arr[0].get("plain_text", "") if title_arr else ""
        
        desc_arr = item.get("description", [])
        desc = desc_arr[0].get("plain_text", "") if desc_arr else ""
        
        return NotionDatabase(
            database_id=item.get("id", ""),
            title=title,
            url=item.get("url", ""),
            description=desc,
            properties=item.get("properties", {}),
        )
