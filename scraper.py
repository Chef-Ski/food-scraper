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

    def __init__(self, csv_filename="sf_food_resources.csv", json_filename="sf_food_resources.json", max_sites=10, max_retries=5, max_depth=10):
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
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--ignore-ssl-errors")
            chrome_options.add_argument("--allow-insecure-localhost")
            
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
            time.sleep(3)  # Give the page more time to load
            
            # Wait for the page to be fully loaded with increased timeout
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning(f"Timeout waiting for page to load: {url}")
                # Continue anyway as we might have partial content
            
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
                elements = soup.find_all(string=re.compile(keyword, re.IGNORECASE))
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
                    if container is None or not hasattr(container, 'name'):
                        break
                    if container.name in ['div', 'section', 'article', 'li', 'table']:
                        break
                    container = container.parent
                
                # Skip if container is None
                if container is None or not hasattr(container, 'name'):
                    continue
                
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
                
            # Ensure all resources are dictionaries before returning
            resources = [r for r in resources if isinstance(r, dict)]
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
            
            # Process in batches to stay within token limits (max 20 entries per batch)
            batch_size = 20
            filtered_data = []
            
            # If very small amount of data, process in one batch
            if len(data) <= batch_size:
                # Create a single batch
                batch_json = json.dumps(data, indent=2)
                
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
                7. IMPORTANT: For entries you keep, preserve ALL original information and field values EXACTLY as they are
                8. DO NOT modify, clean up, or "fix" information in entries you keep - leave them completely intact
                
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
                        try:
                            batch_filtered = list(result_json.values())[0] if result_json else []
                        except (IndexError, TypeError):
                            logger.warning("Unexpected response format from OpenAI, using original batch")
                            batch_filtered = batch
                        
                        # Last resort: if it's just a dict, wrap it in a list
                        if isinstance(batch_filtered, dict):
                            batch_filtered = [batch_filtered]
                        # Ensure we have a list of dictionaries
                        batch_filtered = [item for item in batch_filtered if isinstance(item, dict)]
                    
                    # Log the filtering results
                    logger.info(f"Batch 1: Filtered from {len(batch)} to {len(batch_filtered)} entries")
                    filtered_data.extend(batch_filtered)
                    
                except Exception as e:
                    logger.error(f"Error parsing OpenAI response: {e}")
                    logger.warning(f"Using original batch data due to parsing error")
                    filtered_data.extend(batch)
            else:
                # Process in multiple batches
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
                    7. IMPORTANT: For entries you keep, preserve ALL original information and field values EXACTLY as they are
                    8. DO NOT modify, clean up, or "fix" information in entries you keep - leave them completely intact
                    
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
                            try:
                                batch_filtered = list(result_json.values())[0] if result_json else []
                            except (IndexError, TypeError):
                                logger.warning("Unexpected response format from OpenAI, using original batch")
                                batch_filtered = batch
                            
                            # Last resort: if it's just a dict, wrap it in a list
                            if isinstance(batch_filtered, dict):
                                batch_filtered = [batch_filtered]
                            # Ensure we have a list of dictionaries
                            batch_filtered = [item for item in batch_filtered if isinstance(item, dict)]
                        
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
            target_resources = 150
            
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
                valid_resources = [r for r in resources if isinstance(r, dict) and r.get('Address') and self._validate_address(r.get('Address'))]
                
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
                    self.data = [item for item in self.data if isinstance(item, dict) and item.get('Address') and self._validate_address(item.get('Address'))]
                    
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
            self.data = [item for item in self.data if isinstance(item, dict) and item.get('Address') and self._validate_address(item.get('Address'))]
            
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

    def _count_unique_addresses(self):
        """Count the number of unique addresses in the data."""
        # Filter out non-dictionary items first
        valid_items = [item for item in self.data if isinstance(item, dict) and item.get('Address')]
        
        # Extract and normalize addresses
        addresses = set()
        for item in valid_items:
            addr = item.get('Address', '').strip().lower()
            if addr:
                addresses.add(addr)
                
        return len(addresses)

    def _save_checkpoint(self, url_queue):
        """Save current scraping state to a checkpoint file for resuming later."""
        try:
            # Convert visited_urls set to list for JSON serialization
            visited_list = list(self.visited_urls)
            
            # Convert url_queue to serializable format
            queue_serializable = [[url, depth] for url, depth in url_queue]
            
            # Prepare checkpoint data
            checkpoint_data = {
                'data': [item for item in self.data if isinstance(item, dict)],  # Ensure all items are dictionaries
                'visited_urls': visited_list,
                'sites_scraped': self.sites_scraped,
                'url_depths': self.url_depths,
                'queue': queue_serializable
            }
            
            # Write to checkpoint file
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
                
            logger.info(f"Checkpoint saved: {self.sites_scraped} sites scraped, {len(self.data)} resources found")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _validate_address(self, address):
        """Validate that an address string looks legitimate."""
        if not address:
            return False
            
        # Basic validation - check length and presence of numbers (most addresses have numbers)
        if len(address) < 5:
            return False
            
        # Check for at least one digit (most valid addresses have numbers)
        if not any(char.isdigit() for char in address):
            # Check for PO Box format which might not have street numbers
            if not ('p.o.' in address.lower() or 'po box' in address.lower()):
                return False
                
        # Check for common address words
        address_keywords = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'blvd', 'boulevard', 
                           'drive', 'dr', 'lane', 'ln', 'place', 'pl', 'court', 'ct',
                           'way', 'parkway', 'pkwy', 'highway', 'hwy', 'route', 'box']
                           
        # Convert to lowercase for case-insensitive matching
        address_lower = address.lower()
        
        # Check if the address contains any of the keywords
        has_keyword = any(keyword in address_lower.split() or 
                         keyword+'.' in address_lower or 
                         keyword+',' in address_lower for keyword in address_keywords)
        
        # Also accept if it contains zip code pattern (5 digits together)
        has_zipcode = bool(re.search(r'\b\d{5}\b', address))
        
        return has_keyword or has_zipcode
    
    def retry_failed_urls(self):
        """Retry URLs that failed on first attempt, using exponential backoff."""
        if not self.failed_urls:
            return
            
        logger.info(f"Retrying {len(self.failed_urls)} failed URLs")
        
        # Sort failed URLs by number of attempts (try ones with fewer failures first)
        urls_to_retry = sorted(self.failed_urls.items(), key=lambda x: x[1])
        
        for url, attempts in urls_to_retry:
            # Skip if we've already tried too many times
            if attempts >= self.max_retries:
                logger.warning(f"Skipping {url} - exceeded maximum retry attempts ({attempts})")
                continue
                
            # Skip if already visited successfully in the meantime
            if url in self.visited_urls:
                continue
                
            # Get the original depth if available, otherwise default to 0
            depth = self.url_depths.get(url, 0)
            
            logger.info(f"Retry attempt {attempts+1}/{self.max_retries} for {url}")
            
            # Exponential backoff delay
            wait_time = 2 ** attempts  # 2, 4, 8, 16, 32 seconds
            time.sleep(wait_time)
            
            # Try scraping again
            try:
                resources, _ = self.scrape_page(url, depth=depth)
                
                # If we found valid resources, consider it a success
                valid_resources = [r for r in resources if isinstance(r, dict) and r.get('Address') and self._validate_address(r.get('Address'))]
                if valid_resources:
                    logger.info(f"Retry successful: found {len(valid_resources)} resources at {url}")
                    # Remove from failed URLs (already done in scrape_page if successful)
            except Exception as e:
                logger.error(f"Retry failed for {url}: {e}")

    def _extract_resource_data(self, container, url, source_name=None):
        """Extract structured data from a container element that might have food resource information."""
        if not container:
            return None
            
        # Initialize resource dictionary
        resource = {
            'Source': source_name or urlparse(url).netloc,
            'Source_URL': url,
            'Date_Scraped': time.strftime('%Y-%m-%d')
        }
        
        # Look for name patterns - headers, strong text, etc.
        name_elements = container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b'])
        for element in name_elements:
            # If we find a promising name element
            text = element.get_text().strip()
            if text and 5 < len(text) < 100:  # Reasonable length for a name
                resource['Name'] = text
                break
        
        # If no name found in headers, try to extract from other patterns
        if 'Name' not in resource:
            # Look for spans or divs with "name" class or id
            name_candidates = container.find_all(['span', 'div'], class_=re.compile('name|title', re.I))
            name_candidates += container.find_all(['span', 'div'], id=re.compile('name|title', re.I))
            
            for element in name_candidates:
                text = element.get_text().strip()
                if text and 5 < len(text) < 100:
                    resource['Name'] = text
                    break
            
            # Last resort - use the first reasonable text chunk
            if 'Name' not in resource:
                paragraphs = container.find_all(['p', 'div'])
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and 5 < len(text) < 100:
                        resource['Name'] = text
                        break
        
        # Look for address patterns
        address_candidates = []
        
        # Method 1: Look for common address elements
        address_elements = container.find_all(['address'])
        address_elements += container.find_all(['span', 'div', 'p'], class_=re.compile('address|location', re.I))
        address_elements += container.find_all(['span', 'div', 'p'], id=re.compile('address|location', re.I))
        
        for element in address_elements:
            text = element.get_text().strip()
            if text and self._validate_address(text):
                address_candidates.append(text)
        
        # Method 2: Look for text patterns that look like addresses
        paragraphs = container.find_all(['p', 'div', 'span', 'li'])
        for p in paragraphs:
            text = p.get_text().strip()
            
            # Skip if too short or too long
            if not text or len(text) < 10 or len(text) > 200:
                continue
                
            # Check for address patterns (e.g., "123 Main St")
            if re.search(r'\d+\s+[A-Za-z]+\s+(?:St|Ave|Rd|Blvd|Dr|Lane|Way|Court|Plaza|Square)', text, re.I):
                address_candidates.append(text)
                
            # Check for explicit address labels
            if re.search(r'(?:address|location):\s*(.*)', text, re.I):
                match = re.search(r'(?:address|location):\s*(.*)', text, re.I)
                if match:
                    addr = match.group(1).strip()
                    if addr:
                        address_candidates.append(addr)
        
        # Select the best address candidate
        if address_candidates:
            # Prefer shorter, cleaner addresses
            address_candidates.sort(key=len)
            resource['Address'] = address_candidates[0]
        
        # Look for phone numbers
        phone_pattern = re.compile(r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
        phone_elements = container.find_all(string=phone_pattern)
        
        if phone_elements:
            for element in phone_elements:
                match = phone_pattern.search(element)
                if match:
                    resource['Phone'] = match.group().strip()
                    break
        
        # Look for email addresses
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        email_elements = container.find_all(string=email_pattern)
        
        if email_elements:
            for element in email_elements:
                match = email_pattern.search(element)
                if match:
                    resource['Email'] = match.group().strip()
                    break
        
        # Look for hours of operation
        hours_keywords = ['hours', 'open', 'schedule', 'availability', 'operation']
        hours_elements = []
        
        for keyword in hours_keywords:
            elements = container.find_all(string=re.compile(keyword, re.IGNORECASE))
            for element in elements:
                parent = element.parent
                if parent not in hours_elements:
                    hours_elements.append(parent)
        
        if hours_elements:
            for element in hours_elements:
                text = element.get_text().strip()
                if 'hours' in text.lower() or 'open' in text.lower():
                    # Try to extract just the hours part
                    lines = text.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in hours_keywords):
                            resource['Hours'] = line.strip()
                            break
                    
                    # If not found in lines, use the whole text
                    if 'Hours' not in resource:
                        resource['Hours'] = text
                    break
        
        # If we have at least a name and address, return the resource
        if 'Name' in resource and 'Address' in resource:
            return resource
        
        # If we only have an address but not a name, generate a simple name
        if 'Address' in resource and 'Name' not in resource:
            resource['Name'] = f"Food Resource at {resource['Address']}"
            return resource
            
        return None

    def _extract_generic_content(self, soup, url, source_name=None):
        """
        Fall back method to extract resources when specific extraction fails.
        This uses a more general approach to find content.
        """
        resources = []
        
        # Get the domain name for the source
        source = source_name or urlparse(url).netloc
        
        # Find all paragraphs, list items, and divs that might contain addresses
        elements = soup.find_all(['p', 'li', 'div'])
        
        # Track addresses we've already extracted to avoid duplicates
        found_addresses = set()
        
        for element in elements:
            text = element.get_text().strip()
            
            # Skip elements with too little or too much text
            if not text or len(text) < 15 or len(text) > 500:
                continue
            
            # Look for address patterns
            address_match = re.search(r'\d+\s+[A-Za-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Lane|Ln|Way|Court|Ct|Plaza|Square)', text, re.I)
            
            if address_match:
                # Extract address context (the line containing the address)
                lines = text.split('\n')
                address_line = ""
                
                for line in lines:
                    if address_match.group() in line:
                        address_line = line.strip()
                        break
                
                if not address_line:
                    address_line = address_match.group()
                
                # Skip if we've already found this address
                if address_line in found_addresses:
                    continue
                    
                found_addresses.add(address_line)
                
                # Create a new resource
                resource = {
                    'Source': source,
                    'Source_URL': url,
                    'Date_Scraped': time.strftime('%Y-%m-%d'),
                    'Address': address_line
                }
                
                # Try to extract a name from nearby elements
                name_candidates = []
                
                # Look at previous siblings for potential names
                prev_sibling = element.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b'])
                if prev_sibling:
                    name_text = prev_sibling.get_text().strip()
                    if name_text and 5 < len(name_text) < 100:
                        name_candidates.append(name_text)
                
                # Look at parent's previous siblings
                parent = element.parent
                if parent:
                    prev_parent_sibling = parent.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'b'])
                    if prev_parent_sibling:
                        name_text = prev_parent_sibling.get_text().strip()
                        if name_text and 5 < len(name_text) < 100:
                            name_candidates.append(name_text)
                
                # If we found candidate names, use the closest one
                if name_candidates:
                    resource['Name'] = name_candidates[0]
                else:
                    # Generate a generic name based on content
                    food_terms = [term for term in self.food_keywords if term in text.lower()]
                    if food_terms:
                        resource['Name'] = f"{food_terms[0].title()} at {address_line}"
                    else:
                        resource['Name'] = f"Food Resource at {address_line}"
                
                # Look for phone numbers in the same element
                phone_match = re.search(r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
                if phone_match:
                    resource['Phone'] = phone_match.group().strip()
                
                # Check if this resource has valid data
                if self._validate_address(resource['Address']):
                    resources.append(resource)
        
        return resources

    def _find_relevant_links(self, soup, base_url, current_depth):
        """Find links on the page that are likely to contain food resource information."""
        if current_depth >= self.max_depth:
            return []
            
        next_urls = []
        
        # Get all links from the page
        links = soup.find_all('a', href=True)
        
        # Score and filter links based on relevance
        for link in links:
            href = link.get('href', '')
            
            # Skip empty links, anchors, javascript, etc.
            if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
                
            # Normalize the URL
            full_url = urljoin(base_url, href)
            
            # Skip if we've already visited
            if full_url in self.visited_urls:
                continue
                
            # Skip external links unless they're to a food-related domain
            parsed_base = urlparse(base_url)
            parsed_url = urlparse(full_url)
            
            # Skip links to non-web resources (like PDFs, unless they're specifically about food resources)
            if parsed_url.path.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                text = link.get_text().strip().lower()
                if not any(keyword in text for keyword in self.food_keywords):
                    continue
            
            # Skip different domains unless they look food-related
            if parsed_base.netloc != parsed_url.netloc:
                domain_food_related = any(keyword in parsed_url.netloc for keyword in 
                                        ['food', 'meal', 'pantry', 'bank', 'kitchen', 'feed', 'hunger'])
                                        
                if not domain_food_related:
                    continue
            
            # Calculate relevance score based on link text and URL
            score = 0
            link_text = link.get_text().strip().lower()
            
            # Higher score for links with food-related text
            for keyword in self.food_keywords:
                if keyword in link_text:
                    score += 3
                if keyword in href.lower():
                    score += 2
            
            # Higher score for links that might lead to location information
            for keyword in self.location_keywords:
                if keyword in link_text:
                    score += 2
                if keyword in href.lower():
                    score += 1
            
            # Boost score for specific patterns
            if re.search(r'location|finder|where|map|near|pantry|resource', href.lower()):
                score += 3
                
            # Only consider links with a minimum relevance score
            if score >= 2:
                next_urls.append(full_url)
                
        # Limit the number of links to follow
        next_urls = next_urls[:20]  # Adjust based on how broad you want the crawl to be
        
        return next_urls

    def _delay(self):
        """Add a random delay between requests to avoid overloading servers."""
        delay_time = random.uniform(1.0, 3.0)
        time.sleep(delay_time)

    def _advanced_deduplicate_data(self):
        """Perform advanced deduplication of data using fuzzy matching of addresses."""
        if not self.data:
            return
            
        logger.info(f"Starting advanced deduplication of {len(self.data)} resources")
        
        # First pass: exact address deduplication
        unique_addresses = {}
        
        for item in self.data:
            if not isinstance(item, dict) or 'Address' not in item:
                continue
                
            addr = item.get('Address', '').strip().lower()
            if not addr:
                continue
                
            # Normalize address
            addr = re.sub(r'\s+', ' ', addr)
            
            # If we already have this address, keep the entry with more information
            if addr in unique_addresses:
                existing = unique_addresses[addr]
                
                # Count fields with data in each entry
                existing_fields = sum(1 for v in existing.values() if v)
                item_fields = sum(1 for v in item.values() if v)
                
                # If new item has more information, replace the existing one
                if item_fields > existing_fields:
                    unique_addresses[addr] = item
            else:
                unique_addresses[addr] = item
        
        logger.info(f"After exact address deduplication: {len(unique_addresses)} resources")
        
        # Update self.data with deduplicated data
        self.data = list(unique_addresses.values())
        
        return

# Include only necessary methods to keep file size manageable
# Full implementation would include helper methods like _extract_resource_data, _validate_address, etc.

if __name__ == "__main__":
    # Initialize and run the scraper with increased limits for finding 300+ unique resources
    scraper = FoodResourceScraper(
        max_sites=500, 
        max_depth=10,
        max_retries=5,
        csv_filename="sf_food_resources_expanded.csv",
        json_filename="sf_food_resources_expanded.json"
    )
    scraper.run_scraper()
