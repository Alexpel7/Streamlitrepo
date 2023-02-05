import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


browser = webdriver.Chrome(service = Service(ChromeDriverManager().install()))
browser.get('https://www.chrono24.es/')

#acepta el boton de las cookies
cookies_button = browser.find_element(By.CSS_SELECTOR, 'button.btn.btn-primary.btn-full-width.js-cookie-submit.wt-consent-layer-accept-all')
cookies_button.click()

#pulsa el boton de rolex
#rolex_button = driver.find_element_by_link_text("<a href="/rolex/index.htm" data-ga-event-context="brand-bar-rolex" data-ga-event-label="brand-bar-module" class="d-block bg-sub-high border-radius-large p-x-4 p-y-3 m-b-2">Rolex</a>")
#rolex_button.click()

browser.find_element(By.CSS_SELECTOR, 'a[data-ga-event-context=\'brand-bar-rolex\']').click()