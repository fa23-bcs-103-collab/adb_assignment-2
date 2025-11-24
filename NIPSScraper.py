import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import re
from urllib.parse import urljoin
import time

class NIPSScraper:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="nips_papers"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.collection = self.db["papers_2024"]
        self.base_url = "https://papers.nips.cc/paper_files/paper/2024"
    
    def scrape_papers(self):
        """Scrape all papers from NIPS 2024 website"""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            papers = []
    
            paper_links = soup.find_all('a', href=re.compile(r'/paper/\d+'))
            
            for link in paper_links:
                paper_url = urljoin(self.base_url, link['href'])
                paper_data = self.scrape_paper_details(paper_url)
                if paper_data:
                    papers.append(paper_data)
                time.sleep(1) 
            
            return papers
            
        except Exception as e:
            print(f"Error scraping papers: {e}")
            return []
    
    def scrape_paper_details(self, paper_url):
        """Scrape individual paper details"""
        try:
            response = requests.get(paper_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
   
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text().strip() if title_elem else "Unknown Title"
            
            authors = []
            author_elems = soup.find_all('a', href=re.compile(r'/person/'))
            for author_elem in author_elems:
                authors.append(author_elem.get_text().strip())
            
            paper_data = {
                "title": title,
                "authors": authors,
                "link": paper_url,
                "keywords": self.extract_keywords(title),  # Basic keyword extraction
                "created_at": time.time()
            }
            
            return paper_data
            
        except Exception as e:
            print(f"Error scraping paper {paper_url}: {e}")
            return None
    
    def extract_keywords(self, title):
        """Extract potential keywords from title"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords = [word for word in words if word not in stop_words]
        return keywords
    
    def store_papers(self, papers):
        """Store papers in MongoDB"""
        if papers:
            self.collection.insert_many(papers)
            print(f"Stored {len(papers)} papers in MongoDB")
            
    
            self.create_search_index()
    
    def create_search_index(self):
        """Create Atlas Search index for better search performance"""
        try:
            index_model = SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "title": {
                                "type": "string",
                                "analyzer": "lucene.standard"
                            },
                            "authors": {
                                "type": "string",
                                "analyzer": "lucene.standard"
                            },
                            "keywords": {
                                "type": "string",
                                "analyzer": "lucene.standard"
                            }
                        }
                    }
                },
                name="paper_search"
            )
            self.collection.create_search_index(model=index_model)
        except Exception as e:
            print(f"Note: Atlas Search might not be available: {e}")

# Usage
scraper = NIPSScraper()
papers = scraper.scrape_papers()
scraper.store_papers(papers)