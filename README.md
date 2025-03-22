# Food Resource Scraper

A powerful web scraper designed to collect information about food pantries, soup kitchens, and other food assistance resources across San Francisco and surrounding areas.

## Features

- **Multi-source crawling**: Intelligently follows relevant links from seed URLs to discover food resources
- **Smart data extraction**: Identifies food resource information even in unstructured content
- **Address validation**: Ensures only resources with valid addresses are included
- **Advanced deduplication**: Merges similar entries using name/address matching algorithms
- **Data filtering**: Uses OpenAI to validate real food locations vs. false positives
- **Multiple output formats**: Saves data in both CSV and JSON formats

## Requirements

- Python 3.7+
- Chrome browser (for Selenium WebDriver)
- OpenAI API key (for filtering; optional but recommended)
- Required Python packages (see below)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Chef-Ski/food-scraper.git
   cd food-scraper
   ```

2. Install required packages:
   ```bash
   pip install requests beautifulsoup4 pandas selenium tqdm python-dotenv openai
   ```

3. Download ChromeDriver that matches your Chrome version:
   - Check your Chrome version: Open Chrome and go to Menu > Help > About Google Chrome
   - Download the matching ChromeDriver from [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
   - Add the ChromeDriver to your PATH or specify its location in the script

4. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

Run the scraper with default settings:

```python
python scraper.py
```

This will start the scraping process with the following defaults:
- Max 500 sites to crawl
- Max crawl depth of 4 links
- 5 retry attempts for failed URLs
- Output files: `sf_food_resources_expanded.csv` and `sf_food_resources_expanded.json`

### Customizing the Scraper

You can modify the scraper behavior by editing these parameters in the script:

```python
scraper = FoodResourceScraper(
    max_sites=500,       # Maximum number of sites to scrape
    max_depth=4,         # Maximum depth to follow links
    max_retries=5,       # Maximum retries for failed URLs
    csv_filename="sf_food_resources_expanded.csv",  # CSV output file
    json_filename="sf_food_resources_expanded.json" # JSON output file
)
```

## How It Works

1. **Initialization**: The scraper starts with a list of seed URLs known to contain food resource information
2. **Crawling**: For each page, the scraper:
   - Loads the page using Selenium with headless Chrome
   - Extracts food resource data using various extraction strategies
   - Finds and scores relevant links to follow next
3. **Validation**: Resources without valid addresses are filtered out
4. **Deduplication**: Similar entries are merged to remove duplicates
5. **AI Filtering**: OpenAI is used to filter out non-food resources and invalid entries
6. **Output**: The final dataset is saved in CSV and JSON formats

## Data Fields

The scraper collects the following information when available:

- Resource name
- Description
- Address
- City
- Zip code
- Phone number
- Email address
- Website
- Operating hours
- Food type (Hot Meals, Groceries, Fresh Produce, etc.)
- Services offered
- Eligibility requirements
- Data source information

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.