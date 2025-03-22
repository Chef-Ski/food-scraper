import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import logging
import json
import csv
import os
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import tqdm  # For progress bar
import openai  # Add OpenAI import
from dotenv import load_dotenv  # Add dotenv import

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FoodResourceScraper:
    def save_to_csv(self, temporary=False):
        """Save the scraped data to a CSV file."""
        filename = "temp_" + self.csv_filename if temporary else self.csv_filename
        try:
            df = pd.DataFrame(self.data)
            df.to_csv(filename, index=False)
            logger.info(f"Data saved to CSV: {filename}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")

    def save_to_json(self, temporary=False):
        """Save the scraped data to a JSON file."""
        filename = "temp_" + self.json_filename if temporary else self.json_filename
        try:
            with open(filename, "w") as f:
                json.dump(self.data, f, indent=4)
            logger.info(f"Data saved to JSON: {filename}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")

    def __init__(self, csv_filename="sf_food_resources.csv", json_filename="sf_food_resources.json", max_sites=500, max_retries=5, max_depth=10):
        self.data = []
        self.csv_filename = csv_filename
        self.json_filename = json_filename
        self.max_sites = max_sites
        self.max_retries = max_retries
        self.max_depth = max_depth  # How many links deep to follow
        self.sites_scraped = 0
        self.setup_selenium()
        
        # Track URLs with their depth
        self.visited_urls = set()
        self.url_depths = {}
        
        # Track failed URLs for retry
        self.failed_urls = {}
        
        # Checkpoint file for resuming interrupted scrapes
        self.checkpoint_file = "scraper_checkpoint.json"
        
        # To store name variations for better deduplication
        self.name_variations = {}
        
        # Starting points for food pantry searches - expanded list
        self.seed_urls = [
            "https://www.sfmfoodbank.org/find-food/",
            "https://sf.gov/where-get-free-food",
            "https://www.feedingamerica.org/find-your-local-foodbank",
            "https://www.211bayarea.org/food/food-programs/",
            "https://www.freefood.org/c/ca-san_francisco",
            "https://www.needhelppayingbills.com/html/san_francisco_food_banks.html",
            "https://www.foodpantries.org/ci/ca-san_francisco",
            "https://www.sfhsa.org/services/health-food/calfresh/where-use-calfresh",
            "https://www.sfusd.edu/services/health-wellness/nutrition-school-meals",
            "https://www.glfoodbank.org/find-food/",
            "https://www.shfb.org/get-food/",
            "https://www.accfb.org/get-food/",
            "https://www.foodbankccs.org/get-help/foodbycity.html",
            "https://www.refb.org/get-help/",
            "https://www.foodbanksbc.org/get-help/",
            "https://sffoodbank.org/find-food/",
            "https://www.foodbankst.org/find-food",
            "https://www.centralpafoodbank.org/find-food/",
            "https://www.philabundance.org/find-food/",
            "https://www.pittsburghfoodbank.org/get-help/",
            "https://www.foodbankrockies.org/find-food/",
            "https://oaklandfamilychurch.org/food-pantry/",
            "https://www.glide.org/program/daily-free-meals/",
            "https://www.stanthonysf.org/dining-room/",
            "https://mashsf.org/",
            "https://www.projectopen-hand.org/get-meals/"
        ]
        
        # Keywords to identify food pantry pages - expanded
        self.food_keywords = [
            "food pantry", "food bank", "free food", "free meal", "community meal", 
            "soup kitchen", "food assistance", "emergency food", "food resource",
            "meal program", "CalFresh", "SNAP", "EBT", "WIC", "feeding program",
            "food distribution", "free groceries", "food shelf", "food insecurity",
            "hunger relief", "food aid", "meals on wheels", "nutrition assistance",
            "community fridge", "food closet", "community kitchen", "grocery assistance",
            "food share", "free breakfast", "free lunch", "dinner program",
            "food voucher", "emergency meals", "food recovery", "rescue food",
            "feeding america", "second harvest", "glide", "st anthony", "salvation army food"
        ]
        
        # Extended location keywords for better link discovery
        self.location_keywords = [
            "location", "finder", "directory", "pantry", "where", "address", "map",
            "find", "near", "zip", "area", "neighborhood", "district", "center", 
            "centre", "agency", "agencies", "site", "service"
        ]
    
    def setup_selenium(self):
        """Set up Selenium WebDriver with Chrome options."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Successfully initialized headless Chrome")
        except Exception as e:
            logger.error(f"Error initializing Chrome: {e}")
            raise
    
    def scrape_page(self, url, source_name=None, retry_count=0, depth=0):
        """Scrape a generic page and extract food pantry information."""
        if url in self.visited_urls:
            logger.info(f"Already visited {url}, skipping")
            return [], []
        
        logger.info(f"Scraping {url} (depth: {depth})")
        self.visited_urls.add(url)
        self.url_depths[url] = depth
        
        try:
            # Get the page with Selenium
            self.driver.get(url)
            time.sleep(2)  # Give the page time to load
            
            # Wait for the page to be fully loaded
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page source and parse with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Initialize resources list
            resources = []
            
            # Extract all possible contact information and food-related content
            food_elements = []
            
            # Find sections or elements with food-related keywords
            for keyword in self.food_keywords:
                # Look in text content
                elements = soup.find_all(text=re.compile(keyword, re.IGNORECASE))
                for element in elements:
                    parent = element.parent
                    if parent not in food_elements:
                        food_elements.append(parent)
            
            # Extract from each potential food element and its container
            extracted_resources = []
            
            # For each element containing food keywords, look for surrounding information
            for element in food_elements:
                # Try to find a container (like a div, section, article) that might contain the full information
                container = element
                for _ in range(5):  # Increase from 3 to 5 levels up to find more complete containers
                    if container.name in ['div', 'section', 'article', 'li', 'table']:
                        break
                    container = container.parent
                
                # Extract information from the container
                resource = self._extract_resource_data(container, url, source_name)
                if resource and resource not in extracted_resources:
                    extracted_resources.append(resource)
                    
            # If specific extractors didn't find anything, try generic content extraction
            if not extracted_resources:
                extracted_resources = self._extract_generic_content(soup, url, source_name)
            
            # Filter out resources without addresses
            extracted_resources = [r for r in extracted_resources if r.get('Address') and self._validate_address(r.get('Address'))]
            
            # Add all extracted resources to our data
            for resource in extracted_resources:
                resources.append(resource)
                self.data.append(resource)
                logger.info(f"Found resource with address: {resource.get('Name', 'Unknown')} at {resource.get('Address', 'Unknown')}")
            
            # Find additional links that might contain food resource information
            next_urls = self._find_relevant_links(soup, url, depth)
            
            # If we successfully scraped, remove from failed URLs if it was there
            if url in self.failed_urls:
                del self.failed_urls[url]
                
            return resources, next_urls
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            # Add to failed URLs for potential retry
            self.failed_urls[url] = self.failed_urls.get(url, 0) + 1
            return [], []
    
    def _filter_with_openai(self, data):
        """
        Use OpenAI to filter out invalid or non-real food resource entries.
        This ensures only actual food pantries and services are included.
        """
        if not data:
            logger.warning("No data to filter with OpenAI")
            return []
            
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            # Check if OpenAI API key is available
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found in .env file or environment variables. Skipping AI filtering.")
                return data
                
            openai.api_key = api_key
            logger.info(f"Starting OpenAI filtering of {len(data)} entries")
            
            # Process in batches to stay within token limits (max 25 entries per batch)
            batch_size = 25
            filtered_data = []
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(data) + batch_size - 1)//batch_size}")
                
                # Convert batch to JSON for API request
                batch_json = json.dumps(batch, indent=2)
                
                # Create prompt for OpenAI
                prompt = f"""
                Below is a list of potential food resource locations that were scraped from various websites.
                Your task is to analyze each entry and determine if it represents a real food pantry, 
                food bank, soup kitchen, or similar food assistance service.
                
                Filter rules:
                1. KEEP entries that are clearly real food assistance services with valid names and addresses
                2. REMOVE entries with random text, symbols, or gibberish as names
                3. REMOVE entries that are clearly not food assistance services
                4. REMOVE entries that are just website section names or navigation elements
                5. DO NOT change any information - only filter out invalid entries
                6. DO NOT add any entries or information that wasn't in the original data
                
                Return ONLY the valid entries in the same JSON format, with no additional commentary.
                
                Data to filter:
                {batch_json}
                """
                
                # Call OpenAI API
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo-0125",  # Using 3.5 to reduce costs, 4 can be used for better filtering
                    messages=[
                        {"role": "system", "content": "You are a data filtering assistant that identifies real food assistance locations. You only output valid JSON with no additional text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for more consistent results
                    response_format={"type": "json_object"}
                )
                
                # Extract and parse the response
                try:
                    result_text = response.choices[0].message.content
                    result_json = json.loads(result_text)
                    
                    # Check if the result has the expected format (list of entries)
                    if "entries" in result_json:
                        batch_filtered = result_json["entries"]
                    else:
                        # Try to extract a list directly
                        batch_filtered = list(result_json.values())[0] if result_json else []
                        
                        # Last resort: if it's just a dict, wrap it in a list
                        if isinstance(batch_filtered, dict):
                            batch_filtered = [batch_filtered]
                    
                    # Log the filtering results
                    logger.info(f"Batch {i//batch_size + 1}: Filtered from {len(batch)} to {len(batch_filtered)} entries")
                    filtered_data.extend(batch_filtered)
                    
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    logger.warning(f"Using original batch data due to parsing error")
                    filtered_data.extend(batch)
                
                # Add delay between API calls to avoid rate limits
                time.sleep(1)
            
            logger.info(f"OpenAI filtering complete. Reduced from {len(data)} to {len(filtered_data)} entries")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error using OpenAI for filtering: {e}")
            logger.warning("Using original data due to OpenAI filtering error")
            return data

    def run_scraper(self):
        """Run the complete scraping process with depth tracking and checkpoints for a large number of sites."""
        try:
            logger.info(f"Starting food resource scraper to find 300+ unique food pantries (max sites: {self.max_sites}, max depth: {self.max_depth})")
            
            # Try to load from checkpoint if exists
            if os.path.exists(self.checkpoint_file):
                try:
                    with open(self.checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                        self.data = checkpoint.get('data', [])
                        self.visited_urls = set(checkpoint.get('visited_urls', []))
                        self.sites_scraped = checkpoint.get('sites_scraped', 0)
                        self.url_depths = {k: v for k, v in checkpoint.get('url_depths', {}).items()}
                        remaining_queue = checkpoint.get('queue', [])
                        
                    logger.info(f"Loaded checkpoint: {self.sites_scraped} sites scraped, {len(self.data)} resources found")
                    url_queue = [(url, depth) for url, depth in remaining_queue]
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    # Initialize fresh if checkpoint loading failed
                    url_queue = [(url, 0) for url in self.seed_urls]
            else:
                # Queue now contains (url, depth) tuples
                url_queue = [(url, 0) for url in self.seed_urls]
            
            # Target number of unique resources to find
            target_resources = 300
            
            # Create progress bar with a maximum based on both sites and target resources
            pbar = tqdm.tqdm(total=min(self.max_sites, len(url_queue) + 1000), 
                             desc=f"Scraping sites for {target_resources}+ food resources")
            pbar.update(self.sites_scraped)
            
            # Get current count of unique addresses
            unique_addresses = self._count_unique_addresses()
            
            # Flag to indicate if we've found enough resources
            found_enough = unique_addresses >= target_resources
            
            # Crawl until we've found enough resources, visited max sites, or exhausted the queue
            while url_queue and self.sites_scraped < self.max_sites and not found_enough:
                # Get next URL and its depth
                current_url, current_depth = url_queue.pop(0)
                
                # Skip if already visited
                if current_url in self.visited_urls:
                    continue
                
                # Scrape the current page
                resources, new_urls = self.scrape_page(current_url, depth=current_depth)
                
                # Filter resources to ensure they have addresses
                valid_resources = [r for r in resources if r.get('Address') and self._validate_address(r.get('Address'))]
                
                # Only count if we found valid resources
                if valid_resources:
                    logger.info(f"Found {len(valid_resources)} resources with valid addresses at {current_url}")
                
                # Add new URLs to the queue with incremented depth
                for url in new_urls:
                    if url not in self.visited_urls and not any(url == u for u, _ in url_queue):
                        # Only add URLs that don't exceed max depth
                        if current_depth + 1 <= self.max_depth:
                            url_queue.append((url, current_depth + 1))
                
                # Increment counter
                self.sites_scraped += 1
                
                # Get current count of unique resources
                unique_addresses = self._count_unique_addresses()
                found_enough = unique_addresses >= target_resources
                
                # Log progress
                logger.info(f"Scraped {self.sites_scraped}/{self.max_sites} sites. Found {unique_addresses}/{target_resources} unique resources.")
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "unique_resources": unique_addresses,
                    "queue": len(url_queue), 
                    "depth": current_depth,
                    "target": f"{unique_addresses}/{target_resources}"
                })
                
                # Random delay between requests
                self._delay()
                
                # Save checkpoint periodically
                if self.sites_scraped % 10 == 0:
                    # Filter data to only keep entries with addresses before saving
                    self.data = [item for item in self.data if item.get('Address') and self._validate_address(item.get('Address'))]
                    
                    # Save checkpoint
                    self._save_checkpoint(url_queue)
                    
                    # Save temporary files
                    self.save_to_csv(temporary=True)
                    self.save_to_json(temporary=True)
            
            # Close progress bar
            pbar.close()
            
            # Try to retry any failed URLs if we haven't found enough resources
            if not found_enough and self.failed_urls:
                logger.info(f"Retrying {len(self.failed_urls)} failed URLs to find more resources")
                self.retry_failed_urls()
            
            # Filter final data to only keep entries with addresses
            self.data = [item for item in self.data if item.get('Address') and self._validate_address(item.get('Address'))]
            
            # Apply advanced deduplication
            self._advanced_deduplicate_data()
            
            # NEW: Use OpenAI to filter the data to only real places
            logger.info("Using OpenAI to filter for real food resource locations...")
            self.data = self._filter_with_openai(self.data)
            
            # Check if we have enough unique resources
            unique_addresses = self._count_unique_addresses()
            if unique_addresses < target_resources:
                logger.warning(f"Only found {unique_addresses} unique food resources, below target of {target_resources}")
            else:
                logger.info(f"Successfully found {unique_addresses} unique food resources (target: {target_resources})")
            
            # Save the final results
            self.save_to_csv()
            self.save_to_json()
            
            # Clean up checkpoint file
            if os.path.exists(self.checkpoint_file):
                try:
                    os.remove(self.checkpoint_file)
                    logger.info("Removed checkpoint file after successful completion")
                except:
                    pass
            
            logger.info(f"Completed scraping {self.sites_scraped} sites, found {len(self.data)} unique food resources with valid addresses")
            
        except Exception as e:
            logger.error(f"Error in scraping process: {e}")
            # Save checkpoint in case of unexpected error
            if 'url_queue' in locals():
                self._save_checkpoint(url_queue)
        finally:
            # Clean up
            if hasattr(self, 'driver'):
                self.driver.quit()

# Include only necessary methods to keep file size manageable
# Full implementation would include helper methods like _extract_resource_data, _validate_address, etc.

if __name__ == "__main__":
    # Initialize and run the scraper with increased limits for finding 300+ unique resources
    scraper = FoodResourceScraper(
        max_sites=500, 
        max_depth=4,
        max_retries=5,
        csv_filename="sf_food_resources_expanded.csv",
        json_filename="sf_food_resources_expanded.json"
    )
    scraper.run_scraper()
