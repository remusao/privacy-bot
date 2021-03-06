#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""
Privacy Bot - privacy policies finder.

Usage:
    find_policies [options] [<url>...]

Options:
    -u, --urls U        File containing a list of urls
    -j, --jobs J        Number of parallel jobs [default: 10]
    -h, --help          Show help
"""


import concurrent.futures as futures
import json
import logging
import sys
import docopt
from bs4 import BeautifulSoup

import privacy_bot.fetcher as fetcher

KEYWORDS = ['privacy', 'datenschutz',
            'Конфиденциальность', 'Приватность', 'тайность',
            '隐私', '隱私', 'プライバシー', 'confidential',
            'mentions-legales']

LANGS = {
    'fr': 'fr',
    'de': 'de',
    'com': 'en',
    'co.uk': 'uk',
    'ru': 'ru'
}


def iter_protocols(base_url):
    for protocol in ['https://', 'http://']:
        yield protocol + base_url


def iter_policy_static(url):
    patterns = [
        '/privacy',
        '/privacy-policy',
        '/privacy/privacy-policy',
        '/legal/privacy',
        '/legal/confidential',
    ]
    for p in patterns:
        yield url.rstrip('/') + p


def iter_policy_heuristic(url, fetch):
    """Fetch the page and try to find a privacy URL in it."""
    content = fetch(url)
    if not content:
        return

    soup = BeautifulSoup(content, 'lxml')
    for link in soup.find_all('a', href=True):
        for keyword in KEYWORDS:
            href = link['href']
            href_lower = href.lower()
            text = link.text.lower()
            if keyword in href_lower or keyword in text:
                print(text, href_lower)
                # Get full privacy policy URL
                if href.startswith('//'):
                    href = 'http:' + href
                elif href.startswith('/'):
                    href = url.rstrip('/') + href
                yield href


def iter_second_level_url(url):
    # TODO: Make heuristic
    for p in ['/terms', '/legal', '/policies']:
        yield url.rstrip('/') + p


def iter_url_candidates(base_url, level=0):
    for url in iter_protocols(base_url):
        # Check to extract based on heuristic
        yield from iter_policy_heuristic(url)
        # Check static list
        # yield from iter_policy_static(url) #TODO: remove
        # Check at second level
        # if level == 0:
        #     for second_level_url in iter_second_level_url(base_url):
        #         yield from iter_url_candidates(second_level_url, level=1)


def get_privacy_policy_url(base_url):
    """Given a valid URL, try to locate the privacy statement page. """
    candidates = set()
    for fetch in [fetcher.fetch, fetcher.fetch_headless]:
        for url in iter_protocols(base_url):
            print('Try:', fetch.__name__, url)
            for candidate in iter_policy_heuristic(url, fetch):
                candidates.add(candidate)

            # Stop as soon as we found a valid page and extracted URLs from it.
            if candidates:
                return list(candidates)
    return []


def main():
    logging.basicConfig(level=logging.ERROR)

    args = docopt.docopt(__doc__)
    jobs = int(args['--jobs'])

    # Gather every urls
    urls = args['<url>']
    from_file = args.get('--urls')
    if from_file is not None:
        with open(from_file) as urls_file:
            urls.extend(urls_file)

    # Remove comments and empty lines
    urls = set(
        url.strip() for url in urls
        if not url.startswith('#') and len(url.strip()) > 0
    )

    # Fetch data
    if urls:
        print(">> Privacy Bot",                 file=sys.stderr)
        print('-' * 80,                         file=sys.stderr)
        print("Processing %s urls" % len(urls), file=sys.stderr)
        print("Number of jobs: %s" % jobs,      file=sys.stderr)
        print('-' * 80,                         file=sys.stderr)
        print('',                               file=sys.stderr)

        # Find privacy policies
        with futures.ProcessPoolExecutor(jobs) as pool:
            policies = pool.map(get_privacy_policy_url, urls)

        # Generate policies_metadata file
        print('',                       file=sys.stderr)
        print('Generating policies_metadata file', file=sys.stderr)
        print('-' * 80, file=sys.stderr)

        policies_metadata = {}
        for url, policies in zip(urls, policies):
            print('url:', url, 'policies:', policies)
            policies_metadata[url] = {
                "domain": url,
                "privacy_policies": policies,
                "locale": "fr-FR"
            }

        with open('found_policies.json', 'w') as output:
            json.dump(policies_metadata, output, sort_keys=True, indent=4)

        print('-' * 80, file=sys.stderr)

if __name__ == "__main__":
    main()
