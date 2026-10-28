"""
paperless_client.py – REST client for the Paperless-ngx API.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

class PaperlessClient:
    """Client for the Paperless-ngx API."""

    def __init__(self, base_url: str, token: str, max_retries: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Token {self.token}"})
        self._source_id_field_id: int | None = None

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Shared retry helper for all requests. Retries on 5xx and 429."""
        backoff = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.request(method, url, **kwargs)
                
                # Success or client error (4xx except 429) should be returned/raised immediately
                if response.status_code < 500 and response.status_code != 429:
                    if 400 <= response.status_code < 500:
                        raise requests.HTTPError(
                            f"Client Error ({response.status_code}) for {method} {url}: {response.text}",
                            response=response
                        )
                    return response
            except requests.ConnectionError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Could not connect to Paperless-ngx at {self.base_url}: {e}")
                response = None

            if attempt == self.max_retries:
                if response is not None:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} after {self.max_retries} retries for {method} {url}: {response.text}",
                        response=response
                    )
                else:
                    raise RuntimeError(f"Failed to connect after {self.max_retries} retries.")

            print(f"[PaperlessClient] Retrying {method} {url} in {backoff}s (attempt {attempt+1}/{self.max_retries})...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            
        raise RuntimeError("Unreachable")

    def list_custom_fields(self) -> dict[str, dict]:
        """Returns {name: {id, data_type}} for all custom fields."""
        url = f"{self.base_url}/custom_fields/"
        response = self._request_with_retry("GET", url)
        fields = response.json().get("results", [])
        return {f["name"]: {"id": f["id"], "data_type": f["data_type"]} for f in fields}

    def ensure_custom_fields(self, field_specs: list[tuple[str, str]]) -> dict[str, int]:
        """Ensures the specified custom fields exist.
        
        Args:
            field_specs: List of (name, data_type) tuples.
            
        Returns:
            Dict mapping field name to its ID.
        """
        existing = self.list_custom_fields()
        result = {}
        
        for name, data_type in field_specs:
            if name in existing:
                if existing[name]["data_type"] != data_type:
                    print(f"[PaperlessClient] WARNING: Custom field '{name}' has type '{existing[name]['data_type']}', expected '{data_type}'.")
                result[name] = existing[name]["id"]
            else:
                url = f"{self.base_url}/custom_fields/"
                resp = self._request_with_retry("POST", url, json={"name": name, "data_type": data_type})
                result[name] = resp.json()["id"]
        if "source_id" in result:
            self._source_id_field_id = result["source_id"]
                
        return result

    def upload_document(self, pdf_path: Path, title: str) -> str:
        """Uploads a document and returns the task UUID."""
        url = f"{self.base_url}/documents/post_document/"
        with open(pdf_path, "rb") as f:
            response = self._request_with_retry(
                "POST", 
                url, 
                data={"title": title},
                files={"document": (pdf_path.name, f, "application/pdf")}
            )
        return response.text.strip().strip('"')

    def wait_for_consumption(self, task_uuid: str, timeout: int = 120, poll_interval: int = 2) -> int:
        """Polls a consumption task until SUCCESS or FAILURE. Returns document ID."""
        url = f"{self.base_url}/tasks/?task_id={task_uuid}"
        start = time.time()
        
        while time.time() - start < timeout:
            resp = self._request_with_retry("GET", url)
            results = resp.json()
            if not results:
                time.sleep(poll_interval)
                continue
                
            task = results[0]
            status = task.get("status")
            
            if status == "SUCCESS":
                return task["related_document"]
            elif status == "FAILURE":
                raise RuntimeError(f"Paperless consumption failed: {task.get('result')}")
                
            time.sleep(poll_interval)
            
        raise TimeoutError(f"Task {task_uuid} did not complete within {timeout}s.")

    def set_custom_fields(self, document_id: int, values: dict[str, Any], field_id_map: dict[str, int]) -> None:
        """Sets custom field values for a document."""
        url = f"{self.base_url}/documents/{document_id}/"
        
        custom_fields_payload = []
        for name, value in values.items():
            if name in field_id_map:
                field_id = field_id_map[name]
                # Coerce values appropriately if needed, but the Paperless API generally accepts JSON values directly
                custom_fields_payload.append({"field": field_id, "value": value})
                
        self._request_with_retry("PATCH", url, json={"custom_fields": custom_fields_payload})

    def find_document_by_source_id(self, source_id: str) -> int | None:
        """Searches for a document by its source_id custom field."""
        url = f"{self.base_url}/documents/"
        query = json.dumps(["source_id", "exact", source_id])
        params = {"custom_field_query": query}
        logger.debug("Lookup source_id=%r → params=%r", source_id, params)
        resp = self._request_with_retry("GET", url, params=params)
        data = resp.json()
        logger.debug("Lookup response: count=%s, results=%s",
                    data.get("count"), [r.get("id") for r in data.get("results", [])])
        
        results = data.get("results", [])
        if not results:
            return None

        # Defensive check: confirm the match really has this source_id.
        # Guards against future query operator changes silently breaking the lookup.
        for doc in results:
            for cf in doc.get("custom_fields", []):
                if cf.get("field") == self._source_id_field_id and cf.get("value") == source_id:
                    return doc["id"]
    
        logger.warning(
            "Lookup for source_id=%r returned %d documents but none had matching "
            "custom_field value. Treating as miss.",
            source_id, len(results)
        )
        return None