
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

class PubMedLiterature:

    def __init__(self, download_dir: str = "./data/literature", timeout: int = 60):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=timeout, follow_redirects=True,
                                   headers={"User-Agent": "MIHC-Agent/1.0 (research assistant)"})

    def search(self, query: str, max_results: int = 10) -> List[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = self.client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        logger.info("PubMed search '%s' -> %d PMIDs", query, len(ids))
        return ids

    def _pmid_to_pmc(self, pmids: List[str]) -> Dict[str, str]:
        if not pmids:
            return {}
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        resp = self.client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()
        mapping: Dict[str, str] = {}
        for article in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", resp.text, flags=re.S):
            pmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
            pmc = re.search(r"<ArticleId IdType=\"pmc\">(PMC\d+)</ArticleId>", article)
            if pmid and pmc:
                mapping[pmid.group(1)] = pmc.group(1)
        return mapping

    def _oa_links(self, pmc_ids: List[str]) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        for pmc in pmc_ids:
            try:
                params = {"id": pmc}
                resp = self.client.get(OA_BASE, params=params)
                resp.raise_for_status()
                pdf = re.search(rf'<link format="pdf"[^>]*href="([^"]+)"[^>]*/>', resp.text)
                if pdf:
                    links.append({"pmc_id": pmc, "pdf_url": pdf.group(1)})
                else:
                    logger.info("%s 无开放获取 PDF，跳过", pmc)
            except Exception as exc:
                logger.warning("oa.fcgi %s 失败: %s", pmc, exc)
            time.sleep(0.4)
        return links

    def download(self, query: str, max_results: int = 10,
                 max_download: int = 5) -> List[Dict[str, Any]]:
        pmids = self.search(query, max_results=max_results)
        if not pmids:
            return []
        pmc_map = self._pmid_to_pmc(pmids)
        pmc_ids = list(pmc_map.values())[:max_download]
        links = self._oa_links(pmc_ids)

        downloaded: List[Dict[str, Any]] = []
        for link in links:
            target = self.download_dir / f"{link['pmc_id']}.pdf"
            if target.exists():
                downloaded.append({"file_path": str(target), "source": link["pdf_url"], "pmc_id": link["pmc_id"]})
                continue
            try:
                resp = self.client.get(link["pdf_url"])
                resp.raise_for_status()
                if resp.headers.get("content-type", "").startswith("application/pdf") or resp.content[:4] == b"%PDF":
                    target.write_bytes(resp.content)
                    downloaded.append({"file_path": str(target), "source": link["pdf_url"],
                                       "pmc_id": link["pmc_id"]})
                    logger.info("已下载 %s (%d KB)", link["pmc_id"], len(resp.content) // 1024)
                else:
                    logger.info("%s 返回非 PDF 内容，跳过", link["pmc_id"])
            except Exception as exc:
                logger.warning("下载 %s 失败: %s", link["pmc_id"], exc)
            time.sleep(0.4)
        return downloaded
