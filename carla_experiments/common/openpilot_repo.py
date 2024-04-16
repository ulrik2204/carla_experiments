from pathlib import Path
from sys import version
from typing import Literal, Optional, TypedDict, Union

import requests


class LfsResponse(TypedDict):
    version: str
    oid: str
    size: str


def parse_lfs_response(response: str):
    lines = response.split("\n")
    version = lines[0].split(" ")[1]
    oid = lines[1].split(" ")[1].replace("sha256:", "")
    size = lines[2].split(" ")[1]
    return {
        "version": version,
        "oid": oid,
        "size": size,
    }


def download_github_file(
    repo_owner: str,
    repo_name: str,
    file_path_in_repo: str,
    save_path: Optional[Union[str, Path]] = None,
    saved_at: Literal["github", "git-lfs"] = "github",
    main_branch_name: str = "master",
):
    # fetch lastest commit sha
    repo_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    last_commit_url = f"{repo_url}/commits/{main_branch_name}"
    last_commit_response = requests.get(last_commit_url)
    if not last_commit_response.status_code == 200:
        print(f"Failed to fetch latest commit: HTTP {last_commit_response.status_code}")
        return
    commit_sha = last_commit_response.json()["sha"]
    file_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{commit_sha}/{file_path_in_repo}"

    download_response = requests.get(file_url)
    if not download_response.status_code == 200:
        print(
            f"Failed to download {file_path_in_repo}: HTTP {download_response.status_code}"
        )
        return
    if saved_at == "git-lfs":
        lfs_response = parse_lfs_response(download_response.text)
        ob = {
            "operation": "download",
            "transfer": ["basic"],
            "objects": [
                {
                    "oid": lfs_response["oid"],
                    "size": int(lfs_response["size"]),
                }
            ],
        }
        print("ob", ob)
        stored_json = requests.post(
            f"https://github.com/{repo_owner}/{repo_name}.git/info/lfs/objects/batch",
            json=ob,
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-type": "application/json",
            },
        ).json()
        print("stored_json", stored_json)
        link_href = stored_json["objects"][0]["actions"]["download"]["href"]
        download_response = requests.get(link_href)
    # If save_path is not provided, save the file with the original name in the current directory
    save_path = Path(save_path or file_path_in_repo)
    save_path.touch()
    with open(save_path, "wb") as f:
        f.write(download_response.content)
