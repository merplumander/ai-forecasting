from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import List

from gnews import GNews
from googlenewsdecoder import new_decoderv1
from newspaper.article import Article
from tqdm import tqdm

from src.utils import logger

IRRETRIEVABLE_SITES = [
    "wsj.com",
    "english.alarabiya.net",
    "consilium.europa.eu",
    "abc.net.au",
    "thehill.com",
    "democracynow.org",
    "fifa.com",
    "si.com",
    "aa.com.tr",
    "thestreet.com",
    "newsweek.com",
    "spokesman.com",
    "aninews.in",
    "commonslibrary.parliament.uk",
    "cybernews.com",
    "lineups.com",
    "expressnews.com",
    "news-herald.com",
    "c-span.org/video",
    "investors.com",
    "finance.yahoo.com",  # This site has a “read more” button.
    "metaculus.com",  # newspaper4k cannot parse metaculus pages well
    "houstonchronicle.com",
    "unrwa.org",
    "njspotlightnews.org",
    "crisisgroup.org",
    "vanguardngr.com",  # protected by Cloudflare
    "ahram.org.eg",  # protected by Cloudflare
    "reuters.com",  # blocked by Javascript and CAPTCHA
    "carnegieendowment.org",
    "casino.org",
    "legalsportsreport.com",
    "thehockeynews.com",
    "yna.co.kr",
    "carrefour.com",
    "carnegieeurope.eu",
    "arabianbusiness.com",
    "inc.com",
    "joburg.org.za",
    "timesofindia.indiatimes.com",
    "seekingalpha.com",
    "producer.com",  # protected by Cloudflare
    "oecd.org",
    "almayadeen.net",  # protected by Cloudflare
    "manifold.markets",  # prevent data contamination
    "goodjudgment.com",  # prevent data contamination
    "infer-pub.com",  # prevent data contamination
    "www.gjopen.com",  # prevent data contamination
    "polymarket.com",  # prevent data contamination
    "betting.betfair.com",  # protected by Cloudflare
    "news.com.au",  # blocks crawler
    "predictit.org",  # prevent data contamination
    "atozsports.com",
    "barrons.com",
    "forex.com",
    "www.cnbc.com/quotes",  # stock market data: prevent data contamination
    "montrealgazette.com",
    "bangkokpost.com",
    "editorandpublisher.com",
    "realcleardefense.com",
    "axios.com",
    "mensjournal.com",
    "warriormaven.com",
    "tapinto.net",
    "indianexpress.com",
    "science.org",
    "businessdesk.co.nz",
    "mmanews.com",
    "jdpower.com",
    "hrexchangenetwork.com",
    "arabnews.com",
    "nationalpost.com",
    "bizjournals.com",
    "thejakartapost.com",
]


def get_gnews_articles(
    queries: List[str],
    start_date: date = None,
    end_date: date = None,
    max_results: int = 10,
) -> List[List[dict]]:
    """Get articles from Google News for a list of queries.

    Parameters
    ----------
    queries : List[str]
    start_date : date, optional
         Start date of news, by default None
    end_date : date, optional
        End date of news, by default None
    max_results : int, optional
        How many articles to retrieve per query, by default 10

    Returns
    -------
    List[List[dict]]
        Metadata of articles retrieved from Google News. Each inner list
        contains the articles retrieved for a single query.
    """
    #  return empty set for invalid date ranges.
    if end_date is not None and start_date is not None and end_date < start_date:
        raise ValueError("end_date should be strictly more recent than start_date.")
    start_date = tuple(start_date.timetuple())[:3] if start_date is not None else None
    end_date = tuple(end_date.timetuple())[:3] if end_date is not None else None
    retrieved_articles = []

    def get_articles(i):
        google_news = GNews(
            language="en",
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )
        articles = google_news.get_news(queries[i])
        articles = [
            article
            for article in articles
            if article is not None
            and "publisher" in article
            and "href" in article["publisher"]
            and isinstance(article["url"], str) is True
            and not any(site in article["url"] for site in IRRETRIEVABLE_SITES)
        ]
        # Update each article with the search term that retrieved it
        for article in articles:
            article["search_term"] = queries[i]
        return articles

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        retrieved_articles = list(executor.map(get_articles, range(len(queries))))

    return retrieved_articles


def retrieve_gnews_articles_fulldata(
    retrieved_articles: List[List[dict]],
    num_articles: int = 5,
    length_threshold: int = 200,
) -> List[Article]:
    """Retrieve full text of articles from Google News.

    Parameters
    ----------
    retrieved_articles : List[List[dict]]
        Metadata of articles retrieved from Google News. Each inner list
        contains the articles retrieved for a single query.
    num_articles : int, optional
        How many full articles to retrieve per query., by default 5
    length_threshold : int, optional
        Minimum length of articles, by default 200

    Returns
    -------
    List[Article]
        List of Articles with full text retrieved .
    """
    google_news = GNews()
    fulltext_articles = []
    unique_urls = set()

    for articles_group in retrieved_articles:
        articles_added = 0
        for article in articles_group:
            if articles_added >= num_articles:  # we have enough articles
                break
            if article["url"] in unique_urls:  # duplicated article
                continue
            else:  # new article, add to the set of unique urls
                unique_urls.add(article["url"])

    def get_full_article(url):
        url = new_decoderv1(url)["decoded_url"]
        full_article = google_news.get_full_article(url)
        return full_article

    with ThreadPoolExecutor(max_workers=50) as executor:
        full_articles = list(executor.map(get_full_article, unique_urls))

    for full_article in full_articles:
        if (
            full_article is not None
            and full_article.text
            and full_article.publish_date
            and len(full_article.text) > length_threshold
        ):  # remove short articles
            full_article.search_term = article["search_term"]
            full_article.html = ""  # remove html, useless for us
            fulltext_articles.append(full_article)
            articles_added += 1
    if len(fulltext_articles) < len(retrieved_articles) * num_articles:
        logger.warning(
            f"Could only retrieve {len(fulltext_articles)} out of"
            f" {len(retrieved_articles) * num_articles}"
        )
    return fulltext_articles
