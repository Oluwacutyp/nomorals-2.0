"""Auth plugins. Every entry is a `WebPlugin`; the forge drives them."""

from .amazon import AmazonPlugin
from .ebay import EbayPlugin
from .gmail import GmailPlugin
from .instagram import InstagramPlugin
from .interface import AuthResult, EnumerateResult, LoginResult, SessionData, WebPlugin
from .linkedin import LinkedInPlugin
from .netflix import NetflixPlugin
from .proton import ProtonPlugin
from .reddit import RedditPlugin
from .spotify import SpotifyPlugin
from .tiktok import TikTokPlugin
from .twitter import TwitterPlugin
from .yahoo import YahooPlugin

PLUGINS: dict[str, type[WebPlugin]] = {
    p.name: p for p in (
        AmazonPlugin, EbayPlugin, GmailPlugin, InstagramPlugin,
        LinkedInPlugin, NetflixPlugin, ProtonPlugin, RedditPlugin,
        SpotifyPlugin, TikTokPlugin, TwitterPlugin, YahooPlugin,
    )
}

__all__ = ["PLUGINS", "WebPlugin", "LoginResult", "AuthResult",
           "SessionData", "EnumerateResult"]
