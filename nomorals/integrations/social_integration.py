"""Social media integration: Facebook, Instagram, Threads, Messenger.

Supports:
- Read profiles, posts, comments, stories, feeds, insights
- Publish posts, stories, reels, carousels
- Manage Marketplace listings
- Messenger: read/search/send/react/unsend/edit messages

Usage:
    social = SocialMediaIntegration(account_manager, session_manager)
    
    # Post to Instagram
    await social.post_instagram(
        image_url="https://...", caption="Check this out!", account="bot"
    )
    
    # Read Facebook feed
    posts = await social.get_facebook_feed(account="bot", limit=10)
    
    # Send Messenger message
    await social.send_messenger("user_id", "Hello!", account="bot")
    
    # Post to Threads
    await social.post_threads("Thinking about AI...", account="bot")
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["SocialMediaIntegration", "SocialPost", "SocialProfile"]

_log = get_logger(__name__)

GRAPH_API = "https://graph.facebook.com/v18.0"


@dataclass
class SocialPost:
    post_id: str
    platform: str  # facebook, instagram, threads
    content: str = ""
    media_urls: list[str] = field(default_factory=list)
    author: str = ""
    created_time: str = ""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    url: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id, "platform": self.platform,
            "content": self.content, "author": self.author,
            "likes": self.likes, "comments": self.comments,
        }


@dataclass
class SocialProfile:
    profile_id: str
    platform: str
    username: str = ""
    display_name: str = ""
    bio: str = ""
    followers: int = 0
    following: int = 0
    posts_count: int = 0
    profile_pic_url: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "platform": self.platform,
            "username": self.username, "display_name": self.display_name,
            "followers": self.followers,
        }


class SocialMediaIntegration:
    """Meta social media integration via Graph API."""
    
    def __init__(self, account_manager: AccountManager, session_manager: SessionManager) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        _log.info("Social media integration initialized")
    
    async def _graph_request(
        self, method: str, endpoint: str, account: str,
        data: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make Graph API request."""
        cred = self.account_manager.get_credential("facebook_oauth", account)
        token_data = json.loads(cred.password)
        access_token = token_data["access_token"]
        
        all_params = {"access_token": access_token, **(params or {})}
        query = urllib.parse.urlencode(all_params)
        url = f"{GRAPH_API}/{endpoint}?{query}"
        
        body = None
        if data and method != "GET":
            body = urllib.parse.urlencode(data).encode()
            url = f"{GRAPH_API}/{endpoint}"
            query_with_token = urllib.parse.urlencode({"access_token": access_token})
            url = f"{GRAPH_API}/{endpoint}?{query_with_token}"
        
        req = urllib.request.Request(url, data=body, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            _log.error(f"Graph API error: {e.code} - {error_body}")
            raise
    
    # ── Facebook ─────────────────────────────────────────────────────────────
    
    async def get_facebook_profile(self, account: str) -> SocialProfile:
        """Get Facebook profile."""
        result = await self._graph_request("GET", "me", account,
            params={"fields": "id,name,picture,bio"})
        
        return SocialProfile(
            profile_id=result.get("id", ""),
            platform="facebook",
            display_name=result.get("name", ""),
            bio=result.get("bio", ""),
        )
    
    async def get_facebook_feed(self, account: str, *, limit: int = 10) -> list[SocialPost]:
        """Get Facebook feed."""
        result = await self._graph_request("GET", "me/feed", account,
            params={"fields": "id,message,created_time,likes.summary(true),comments.summary(true),shares", "limit": str(limit)})
        
        posts = []
        for item in result.get("data", []):
            posts.append(SocialPost(
                post_id=item.get("id", ""),
                platform="facebook",
                content=item.get("message", ""),
                created_time=item.get("created_time", ""),
                likes=item.get("likes", {}).get("summary", {}).get("total_count", 0),
                comments=item.get("comments", {}).get("summary", {}).get("total_count", 0),
                shares=item.get("shares", {}).get("count", 0),
            ))
        
        return posts
    
    async def post_facebook(
        self, message: str, account: str, *,
        link: str = "", photo_url: str = "",
    ) -> str:
        """Post to Facebook timeline."""
        data: dict[str, Any] = {"message": message}
        if link:
            data["link"] = link
        
        result = await self._graph_request("POST", "me/feed", account, data=data)
        return result.get("id", "")
    
    # ── Instagram ────────────────────────────────────────────────────────────
    
    async def get_instagram_profile(self, account: str) -> SocialProfile:
        """Get Instagram business profile."""
        # First get IG user ID from Facebook page
        result = await self._graph_request("GET", "me", account,
            params={"fields": "instagram_business_account"})
        
        ig_id = result.get("instagram_business_account", {}).get("id", "")
        if not ig_id:
            raise ValueError("No Instagram business account linked")
        
        ig_result = await self._graph_request("GET", ig_id, account,
            params={"fields": "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url"})
        
        return SocialProfile(
            profile_id=ig_id,
            platform="instagram",
            username=ig_result.get("username", ""),
            display_name=ig_result.get("name", ""),
            bio=ig_result.get("biography", ""),
            followers=ig_result.get("followers_count", 0),
            following=ig_result.get("follows_count", 0),
            posts_count=ig_result.get("media_count", 0),
            profile_pic_url=ig_result.get("profile_picture_url", ""),
        )
    
    async def post_instagram(
        self, image_url: str, caption: str, account: str,
    ) -> str:
        """Post image to Instagram (requires business account)."""
        # Get IG user ID
        result = await self._graph_request("GET", "me", account,
            params={"fields": "instagram_business_account"})
        ig_id = result.get("instagram_business_account", {}).get("id", "")
        
        if not ig_id:
            raise ValueError("No Instagram business account linked")
        
        # Create media container
        container = await self._graph_request("POST", f"{ig_id}/media", account,
            data={"image_url": image_url, "caption": caption})
        container_id = container.get("id", "")
        
        # Publish
        result = await self._graph_request("POST", f"{ig_id}/media_publish", account,
            data={"creation_id": container_id})
        
        return result.get("id", "")
    
    async def post_instagram_carousel(
        self, image_urls: list[str], caption: str, account: str,
    ) -> str:
        """Post carousel to Instagram."""
        result = await self._graph_request("GET", "me", account,
            params={"fields": "instagram_business_account"})
        ig_id = result.get("instagram_business_account", {}).get("id", "")
        
        # Create child containers
        child_ids = []
        for url in image_urls:
            child = await self._graph_request("POST", f"{ig_id}/media", account,
                data={"image_url": url, "is_carousel_item": "true"})
            child_ids.append(child.get("id", ""))
        
        # Create carousel container
        carousel = await self._graph_request("POST", f"{ig_id}/media", account,
            data={"caption": caption, "media_type": "CAROUSEL", "children": ",".join(child_ids)})
        
        # Publish
        result = await self._graph_request("POST", f"{ig_id}/media_publish", account,
            data={"creation_id": carousel.get("id", "")})
        
        return result.get("id", "")
    
    # ── Threads ──────────────────────────────────────────────────────────────
    
    async def post_threads(self, text: str, account: str) -> str:
        """Post to Threads."""
        result = await self._graph_request("GET", "me", account,
            params={"fields": "threads_id"})
        threads_id = result.get("threads_id", "")
        
        if not threads_id:
            raise ValueError("No Threads account linked")
        
        # Create container
        container = await self._graph_request("POST", f"{threads_id}/threads", account,
            data={"text": text, "media_type": "TEXT"})
        
        # Publish
        result = await self._graph_request("POST", f"{threads_id}/threads_publish", account,
            data={"creation_id": container.get("id", "")})
        
        return result.get("id", "")
    
    async def get_threads_feed(self, account: str, *, limit: int = 10) -> list[SocialPost]:
        """Get Threads feed."""
        result = await self._graph_request("GET", "me", account,
            params={"fields": "threads_id"})
        threads_id = result.get("threads_id", "")
        
        feed = await self._graph_request("GET", f"{threads_id}/threads", account,
            params={"fields": "id,text,timestamp,like_count,reply_count", "limit": str(limit)})
        
        return [
            SocialPost(
                post_id=item.get("id", ""),
                platform="threads",
                content=item.get("text", ""),
                created_time=item.get("timestamp", ""),
                likes=item.get("like_count", 0),
                comments=item.get("reply_count", 0),
            )
            for item in feed.get("data", [])
        ]
    
    # ── Messenger ────────────────────────────────────────────────────────────
    
    async def send_messenger(self, recipient_id: str, message: str, account: str) -> str:
        """Send a Messenger message."""
        data = {
            "recipient": {"id": recipient_id},
            "message": {"text": message},
        }
        
        result = await self._graph_request("POST", "me/messages", account, data=data)
        return result.get("message_id", "")
    
    async def get_messenger_conversations(self, account: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Get Messenger conversations."""
        result = await self._graph_request("GET", "me/conversations", account,
            params={"fields": "id,messages{message,created_time,from},participants", "limit": str(limit)})
        
        return result.get("data", [])
    
    async def search_messenger(self, query: str, account: str) -> list[dict[str, Any]]:
        """Search Messenger conversations."""
        result = await self._graph_request("GET", "me/conversations", account,
            params={"fields": "id,messages{message,created_time,from}", "limit": "50"})
        
        # Filter by query
        matches = []
        query_lower = query.lower()
        for conv in result.get("data", []):
            for msg in conv.get("messages", {}).get("data", []):
                if query_lower in msg.get("message", "").lower():
                    matches.append({
                        "conversation_id": conv["id"],
                        "message": msg.get("message", ""),
                        "from": msg.get("from", {}).get("name", ""),
                        "created_time": msg.get("created_time", ""),
                    })
        
        return matches
    
    # ── Insights ─────────────────────────────────────────────────────────────
    
    async def get_insights(
        self, page_id: str, account: str, *,
        metrics: list[str] | None = None, period: str = "day",
    ) -> dict[str, Any]:
        """Get page insights."""
        metrics = metrics or ["page_impressions", "page_engaged_users", "page_fans"]
        
        result = await self._graph_request("GET", f"{page_id}/insights", account,
            params={"metric": ",".join(metrics), "period": period})
        
        insights = {}
        for item in result.get("data", []):
            name = item.get("name", "")
            values = item.get("values", [])
            if values:
                insights[name] = values[-1].get("value", 0)
        
        return insights
    
    # ── Marketplace ──────────────────────────────────────────────────────────
    
    async def list_marketplace(
        self, title: str, price: float, description: str, account: str, *,
        images: list[str] | None = None, location: str = "",
    ) -> str:
        """Create a Marketplace listing."""
        data: dict[str, Any] = {
            "name": title,
            "price": str(int(price * 100)),  # In cents
            "description": description,
            "currency": "USD",
        }
        
        if location:
            data["location"] = location
        
        result = await self._graph_request("POST", "me/marketplace_listings", account, data=data)
        return result.get("id", "")
