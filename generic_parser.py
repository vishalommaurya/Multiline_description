"""
Generic Multiline Bank Statement Parser

This module provides a reusable `GenericMultilineParser` class that implements 
robust "Vertical Proximity" and "Line Grouping" logic. It is designed to be 
configurable for any bank that produces PDF statements with:
1. Multiline descriptions
2. Columnar layouts (header-based or fixed)
3. Date-based transaction rows
"""

import re
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field
import pdfplumber

@dataclass
class BankConfig:
    """Configuration for a specific bank's PDF format."""
    bank_name: str
    
    # regex patterns to identify valid header rows
    # Keys should match standard fields: 'date', 'details', 'debit', 'credit', 'balance', 'chq_no'
    # Values are lists of keywords to look for in the header line.
    header_keywords: Dict[str, List[str]]
    
    # Fallback X-coordinates if dynamic header detection fails
    fallback_coords: Dict[str, Any]
    
    # If True, always use fallback_coords for columns instead of detected ones, 
    # but still use detected header_top to skip junk lines above headers.
    force_fallback: bool = False
    
    # Date formats to try parsing
    date_formats: List[str] = field(default_factory=lambda: ['%d/%m/%y', '%d/%m/%Y', '%d-%m-%y', '%d-%m-%Y'])
    
    # Keywords that indicate non-transactional metadata lines (headers, footers, etc.)
    # These will be strictly ignored during parsing.
    footer_markers: List[str] = field(default_factory=lambda: [
        'closing balance', 'page total', 'carried forward', 'brought forward', 
        'statement summary', 'end of statement', 'page no', 'statement time'
    ])
    
    # Regex pattern for money (e.g., "1,234.00", "1234.00Cr")
    money_pattern: str = r'^[\d,.]+(Cr|Dr)?$'
    
    # Thresholds
    header_tolerance_lines: int = 5  # Number of lines to scan for header
    vertical_proximity_ratio: float = 1.2 # If gap_above > ratio * gap_below, it's a multiline part
    
    # If True, allows date detection anywhere on the line (not just left zone)
    # Useful for lines like "UsOnus ... txn dt 02/11/2017" which are valid headers
    loose_date_detection: bool = False
    

class GenericMultilineParser:
    def __init__(self, config: BankConfig):
        self.config = config

    def parse(self, pdf_path: str, password: str = None) -> Tuple[pd.DataFrame, Tuple[str, str]]:
        """Main entry point to parse a PDF."""
        try:
            print(f"[DEBUG generic_parser] Starting {self.config.bank_name} extraction for: {pdf_path}")
            all_transactions = []
            account_name = "_"
            account_number = "XXXXXXXXXX"
            opening_balance = 0.0
            col_coords = None
            current_transaction = None
            
            with pdfplumber.open(pdf_path, password=password) as pdf:
                # OPTIONAL: Extract global metadata from page 0 if needed (could be config-driven too)
                if pdf.pages:
                    first_page_text = pdf.pages[0].extract_text()
                    # Placeholder for custom metadata extraction if passed in config, 
                    # otherwise user can implement headers separately.
                    # For now we'll stick to returning placeholders or basic extraction.
                
                for page_num, page in enumerate(pdf.pages):
                    print(f"[DEBUG generic_parser] Processing page {page_num + 1}")
                    words = page.extract_words(keep_blank_chars=True, x_tolerance=2, y_tolerance=2)
                    if not words: continue
                    
                    full_page_lines = self._group_into_lines(words)
                    global_footer_top = self._find_footer_boundary(full_page_lines)
                    
                    # Process the page
                    header_data = self._find_column_coordinates(full_page_lines)
                    header_top = 0
                    if header_data:
                        header_top = header_data['top']
                        if not self.config.force_fallback:
                            col_coords = header_data['coords']
                    
                    if not col_coords:
                        col_coords = self.config.fallback_coords
                    
                    page_transactions, current_transaction = self._extract_transactions_from_lines(
                        full_page_lines, col_coords, current_transaction, header_top, global_footer_top
                    )
                    all_transactions.extend(page_transactions)
            
            if current_transaction:
                all_transactions.append(current_transaction)
                
            # Filter out junk/empty transactions (no description AND no amounts)
            valid_transactions = []
            for tx in all_transactions:
                has_desc = bool(tx.get('Description', '').strip())
                has_amt = tx.get('Debit', 0) > 0 or tx.get('Credit', 0) > 0
                if has_desc or has_amt:
                    valid_transactions.append(tx)
            
            df = pd.DataFrame(valid_transactions)
            df = self._clean_dataframe(df)
                    
            print(f"[DEBUG generic_parser] Successfully extracted {len(df)} transactions")
            return df, (account_name, account_number)
            
        except Exception as e:
            print(f"[DEBUG generic_parser] Error in {self.config.bank_name} parser: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), ("_", "XXXXXXXXXX")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard cleanup of the resulting DataFrame."""
        if df.empty: return df
        if 'Debit' in df.columns: df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        if 'Credit' in df.columns: df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        if 'Balance' in df.columns: df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)
        
        # Remove internal metadata columns
        cols_to_drop = ['Value Date Top', 'last_y']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return df

    def _group_into_lines(self, words: List[Dict], tolerance: float = 5.0) -> List[List[Dict]]:
        """Group words into lines based on Y-coordinate."""
        if not words: return []
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        lines, current_line, current_y = [], [], None
        y_tolerance = tolerance
        for word in sorted_words:
            word_y = word['top']
            if current_y is None:
                current_y, current_line = word_y, [word]
            elif abs(word_y - current_y) <= y_tolerance:
                current_line.append(word)
            else:
                if current_line: lines.append(current_line)
                current_y, current_line = word_y, [word]
        if current_line: lines.append(current_line)
        return lines

    def _find_column_coordinates(self, lines: List[List[Dict]]) -> Optional[Dict[str, Any]]:
        """Find column X-coordinates and ranges by analyzing the header row."""
        # Flatten simple keywords for initial check
        all_keywords = [k for sublist in self.config.header_keywords.values() for k in sublist]
        
        for line in lines:
            if not line: continue
            line_text = ' '.join([w['text'] for w in line]).lower()
            
            # Check if this line looks like a header (contains at least 3 keywords)
            # Use whole word matching to avoid partial matches
            matches = []
            for col_name, keywords in self.config.header_keywords.items():
                if any(re.search(r'\b' + re.escape(k) + r'\b', line_text) for k in keywords):
                    matches.append(col_name)
            
            match_count = len(set(matches))
            if match_count >= 3:
                header_items = {}
                line_top = line[0]['top']
                
                # Assign words to columns
                for word in line:
                    text_lower = word['text'].lower()
                    for col_name, keywords in self.config.header_keywords.items():
                        if any(k in text_lower for k in keywords):
                            if col_name not in header_items:
                                header_items[col_name] = (word['x0'], word['x1'])
                
                # Logic to convert detected bounding boxes to useful specific coordinates
                coords = {}
                
                # Date boundary (end of date column)
                if 'date' in header_items:
                    coords['date_end_boundary'] = header_items['date'][1] + 5
                else:
                    coords['date_end_boundary'] = self.config.fallback_coords.get('date_end_boundary', 110)

                # Details limit (start of first numeric column usually)
                numeric_cols = ['chq_no', 'debit', 'credit', 'balance']
                first_numeric_x = 1000
                for key in numeric_cols:
                    if key in header_items:
                        start_x = header_items[key][0]
                        if start_x < first_numeric_x:
                            first_numeric_x = start_x
                
                if first_numeric_x < 1000:
                    coords['details_end'] = first_numeric_x - 5
                else:
                    coords['details_end'] = self.config.fallback_coords.get('details_end', 450)
                
                # Details start (after date usually)
                if 'date' in header_items:
                     coords['details_start'] = header_items['date'][1] + 5
                else:
                     coords['details_start'] = self.config.fallback_coords.get('details_start', 100)
                
                # Center points for amount columns
                for key in ['debit', 'credit', 'balance', 'amount']:
                    if key in header_items:
                        coords[f'{key}_x'] = (header_items[key][0] + header_items[key][1]) / 2
                
                return {'coords': coords, 'top': line_top}
        return None

    def _find_footer_boundary(self, lines: List[List[Dict]]) -> float:
        """Find the highest Y-coordinate where a footer block starts."""
        footer_y = 1000.0
        for line in lines:
            text, top = ' '.join([w['text'] for w in line]).lower(), line[0]['top']
            if top < 400: continue # Footers are usually at the bottom
            if any(marker in text for marker in self.config.footer_markers):
                if top < footer_y:
                    footer_y = top
        return footer_y - 2

    def _extract_transactions_from_lines(self, lines: List[List[Dict]], col_coords: Dict[str, Any], 
                                       current_transaction: Optional[Dict] = None, 
                                       header_top: float = 0, footer_top: float = 1000) -> Tuple[List[Dict], Optional[Dict]]:
        """Extract transactions using Proximity approach."""
        transactions, prefix_buffer = [], []
        details_limit = col_coords.get('details_end', 445)
        details_start = col_coords.get('details_start', 100)
        # Safety cap
        if details_limit > 550: details_limit = 445
        
        money_re = re.compile(self.config.money_pattern, re.I)

        for idx, line in enumerate(lines):
            if not line: continue
            line_y = line[0]['top']
            if line_y <= header_top + 2 or line_y >= footer_top: continue

            # Horizontal Truncation (Footer keyword mid-line)
            cleaned_line = []
            for word in line:
                if any(marker in word['text'].lower() for marker in self.config.footer_markers):
                    break
                cleaned_line.append(word)
            if not cleaned_line: continue
            line = cleaned_line
            line_text = ' '.join([w['text'] for w in line])

            # Header Guard
            if self._is_metadata_row(line_text): continue

            # Check for Date (Start of Transaction)
            is_date_row = False
            date_match = self._find_date_in_text(line_text)
            
            # Additional check: Date must be on the left side (Date Zone)
            if date_match:
                if self.config.loose_date_detection:
                    is_date_row = True
                else:    
                    date_zone_limit = max(150, details_start + 10)
                    for word in line:
                        if word['x0'] < date_zone_limit and self._find_date_in_text(word['text']):
                            is_date_row = True; break

            if is_date_row:
                # STRICT BOUNDARY: Immediately flush previous transaction when new date is found
                # This ensures transactions NEVER merge, even if amounts appear on same line
                if current_transaction:
                    # Flush the current transaction (even if incomplete)
                    transactions.append(current_transaction)
                    current_transaction = None
                
                parsed_date = self._parse_date(date_match.group(0).strip())
                
                # Handle Prefix Buffer (lines found before the date that belong to this txn)
                desc_parts = prefix_buffer; prefix_buffer = []
                
                # Extract description and amounts
                # Date indices: find all words that look like dates in the left-hand zone
                date_zone_end = max(180, details_start + 50)
                date_indices = [i for i, word in enumerate(line) if word['x0'] < date_zone_end and self._find_date_in_text(word['text'])]
                
                for i, word in enumerate(line):
                    clean_text = word['text'].strip()
                    if not clean_text: continue
                    
                    # Skip the date itself (and ANY date in the header zone)
                    if i in date_indices:
                        # If the word is mostly a date but has attached text, keep the text
                        # e.g., "01/01/2024NARRATION"
                        text_no_date = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', clean_text).strip()
                        if text_no_date and word['x0'] >= details_start - 10:
                            desc_parts.append(text_no_date)
                        continue

                    if word['x0'] < details_start - 10: continue
                        
                    # Skip Columns Area (Amounts) - Strict Filtering
                    if word['x0'] >= details_limit - 15:
                        # Allow long reference numbers typically found in desc
                        is_long_ref = len(clean_text.replace(',', '')) > 10 and bool(re.search(r'\d{6,}', clean_text))
                        if (money_re.match(clean_text) and not is_long_ref) or clean_text.lower() in ['cr', 'dr', 'cr.', 'dr.']: 
                            continue
                            
                    if word['x0'] > details_limit + 50: continue
                    desc_parts.append(word['text'])

                # Create NEW transaction
                current_transaction = {
                    'Value Date': parsed_date, 'Value Date Top': line_y,
                    'Description': ' '.join(desc_parts).strip(),
                    'Debit': 0.0, 'Credit': 0.0, 'Balance': 0.0,
                    'last_y': line_y
                }
                self._update_transaction_amounts(line, current_transaction, col_coords)
            else:
                # NO DATE ROW - Decide if it fits previous or next
                if self._contains_amount(line, col_coords):
                    # Line has money? 
                    # CRITICAL: If current transaction ALREADY has an amount (on a different line), 
                    # and we find a NEW amount line without a date, it's likely a NEW transaction 
                    # with the same date as the previous one (date was not repeated in PDF).
                    if current_transaction and (current_transaction['Debit'] > 0 or current_transaction['Credit'] > 0):
                        if line_y > current_transaction['last_y'] + 2:
                            # Start NEW transaction on same date
                            transactions.append(current_transaction)
                            prev_date = current_transaction['Value Date']
                            prev_date_top = current_transaction['Value Date Top']
                            
                            current_transaction = {
                                'Value Date': prev_date, 'Value Date Top': prev_date_top,
                                'Description': '', 'Debit': 0.0, 'Credit': 0.0, 'Balance': 0.0,
                                'last_y': line_y
                            }
                    
                    is_closer_to_next = False
                else:
                    # Logic: "Gap Splitter"
                    prev_line_bottom = current_transaction['last_y'] if current_transaction else (lines[idx-1][-1]['top'] if idx > 0 else 0)
                    
                    next_line_top = None
                    next_date_top = None
                    if idx + 1 < len(lines):
                        next_line_top = lines[idx+1][0]['top']
                    
                    # Find next date row
                    for j in range(idx + 1, len(lines)):
                        if self._contains_date_column(lines[j]):
                            next_date_top = lines[j][0]['top']
                            break
                    
                    current_top = line_y
                    gap_above = current_top - prev_line_bottom if prev_line_bottom > 0 else 0
                    gap_below = next_line_top - current_top if next_line_top else 999
                    is_closer_to_next = False

                    if gap_above > self.config.vertical_proximity_ratio * gap_below and gap_below < 30:
                        is_closer_to_next = True
                    elif next_date_top and current_transaction:
                        # If we are geometrically closer to the NEXT date line than the CURRENT date line
                        current_date_top = current_transaction['Value Date Top']
                        dist_to_prev = abs(current_top - current_date_top)
                        dist_to_next = abs(current_top - next_date_top)
                        if dist_to_next * self.config.vertical_proximity_ratio < dist_to_prev:
                            is_closer_to_next = True

                if is_closer_to_next:
                    # Treat as prefix for the UPCOMING transaction
                    line_parts = [word['text'] for word in line if details_start - 10 < word['x0'] < details_limit]
                    prefix_text = ' '.join(line_parts).strip()
                    if prefix_text and not self._is_metadata_row(prefix_text): 
                        prefix_buffer.append(prefix_text)
                else:
                    # This line belongs to the CURRENT transaction (or is orphaned if no transaction exists yet)
                    if current_transaction:
                        # Append to CURRENT transaction (suffix/continuation)
                        desc_parts = []
                        for word in line:
                            clean_text = word['text'].strip()
                            if not clean_text: continue
                            if word['x0'] >= details_limit - 15:
                                 # Same strict check for amounts
                                 if not (len(clean_text.replace(',', '')) > 10 and bool(re.search(r'\d{6,}', clean_text))):
                                     continue
                            if word['x1'] > details_start - 10: # Skip left margin garbage
                                 desc_parts.append(word['text'])
                        continuation_text = ' '.join(desc_parts).strip()
                        if continuation_text:
                            current_transaction['Description'] = (current_transaction['Description'] + ' ' + continuation_text).strip()
                            current_transaction['last_y'] = line_y
                        self._update_transaction_amounts(line, current_transaction, col_coords)
                    else:
                        # NO current transaction yet, but this line is NOT closer to next
                        # This is an ORPHANED prefix line (appears before first date)
                        # Store it in prefix_buffer for the NEXT date we encounter
                        line_parts = [word['text'] for word in line if details_start - 10 < word['x0'] < details_limit]
                        prefix_text = ' '.join(line_parts).strip()
                        if prefix_text and not self._is_metadata_row(prefix_text):
                            prefix_buffer.append(prefix_text)

        return transactions, current_transaction

    # --- Helper Methods ---

    def _find_date_in_text(self, text: str):
        """Finds any date matching supported formats."""
        # Generic regex for date-like things (dd/mm/yyyy or dd-mm-yyyy)
        # We try to be broad here, _parse_date verifies validity
        pattern = r'\d{2}[/-]\d{2}[/-]\d{2,4}'
        return re.search(pattern, text)

    def _contains_date_column(self, line: List[Dict]) -> bool:
        """Check if a line contains a date in the left zone."""
        # Generic limit, use roughly 150 or similar
        return any(word['x0'] < 150 and self._find_date_in_text(word['text']) for word in line)

    def _contains_amount(self, line: List[Dict], col_coords: Dict[str, Any]) -> bool:
        """Check if a line contains any amount in designated columns."""
        for key in ['debit_x', 'credit_x', 'balance_x']:
            if key in col_coords:
                if self._extract_amount_from_column(line, col_coords[key]) > 0:
                    return True
        return False

    def _update_transaction_amounts(self, line: List[Dict], transaction: Dict, col_coords: Dict[str, Any]):
        """Scans line for amounts in relevant columns and updates the transaction."""
        line_text_lower = ' '.join([w['text'] for w in line]).lower()
        
        # 1. Handle separate Debit/Credit columns
        for col_name in ['Debit', 'Credit', 'Balance']:
            key_map = {'Debit': 'debit_x', 'Credit': 'credit_x', 'Balance': 'balance_x'}
            if key_map[col_name] in col_coords:
                val = self._extract_amount_from_column(line, col_coords[key_map[col_name]])
                if val > 0:
                    transaction[col_name] = val
                    
        # 2. Handle single "Amount" column (Common in Kotak and others)
        if 'amount_x' in col_coords:
            amount_val = self._extract_amount_from_column(line, col_coords['amount_x'])
            if amount_val > 0:
                # Look for DR/CR markers in the line text to classify the amount
                if ' dr' in line_text_lower or 'dr.' in line_text_lower:
                    transaction['Debit'] = amount_val
                elif ' cr' in line_text_lower or 'cr.' in line_text_lower:
                    transaction['Credit'] = amount_val
                else:
                    # Default if no marker found (fallback)
                    # If we already have a debit or credit from separate cols, don't overwrite
                    if transaction.get('Debit', 0) == 0 and transaction.get('Credit', 0) == 0:
                        transaction['Debit'] = amount_val # Default to debit or keep as is?

    def _extract_amount_from_column(self, line: List[Dict], column_x: float) -> float:
        """Extract amount from a specific column X-coordinate."""
        for word in line:
            # Basic sanity range check
            if word['x1'] < 300: continue 
            
            # Distance check
            if abs(((word['x0'] + word['x1']) / 2) - column_x) <= 35:
                # CRITICAL: Check if it's a date before parsing as amount
                # "02/11/2017" can be parsed as 2112017.0 if mistakenly checked
                if self._find_date_in_text(word['text']):
                    continue
                amount = self._parse_amount(word['text'])
                if amount > 0: return amount
        return 0.0

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Try to parse date using configured formats."""
        date_str = date_str.strip()
        for fmt in self.config.date_formats:
            try: return datetime.strptime(date_str, fmt)
            except ValueError: continue
        return None

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string."""
        try: 
            clean = re.sub(r'[^\d.]', '', amount_str.replace(',', ''))
            return float(clean)
        except: return 0.0

    def _is_metadata_row(self, line_text: str) -> bool:
        """Check if line is a header row or non-transactional metadata."""
        if not line_text or len(line_text.strip()) < 3: return True
        text_lower = line_text.lower()
        
        # Check against footer markers
        if any(m in text_lower for m in self.config.footer_markers):
            return True
            
        # Check against ALL header keywords
        matches = []
        for col_name, keywords in self.config.header_keywords.items():
            if any(re.search(r'\b' + re.escape(k) + r'\b', text_lower) for k in keywords):
                matches.append(col_name)
        
        if len(set(matches)) >= 3: return True
        
        return False
