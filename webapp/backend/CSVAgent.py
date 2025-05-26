from agno.agent import Agent
from agno.models.ollama import Ollama
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from agno.utils.log import log_debug, log_info, logger
import csv
import json
from textwrap import dedent
from agno.tools import Toolkit
from pathlib import Path
from agno.agent import Agent
from agno.models.ollama import Ollama
import difflib
import smtplib
from email.message import EmailMessage
import ssl
from difflib import get_close_matches
from dotenv import load_dotenv
import os
load_dotenv()

def email_send(body):
    email_sender = os.getenv("EMAIL_SENDER")
    email_receiver = os.getenv("EMAIL_RECEIVER")
    email_password = os.getenv("EMAIL_PASSWORD")

    # email_password = os.environ.get("EMAIL_PASSWORD")

    subject="Warning Mail"
    # body="This is a test email sent from Python by Kai"
    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())

class CSVCustomTools(Toolkit):
    def __init__(
        self,
        csvs: Optional[List[Union[str, Path]]] = None, max_threshold: int = 100, min_threshold: int = 5, **kwargs,
        
    ):
        super().__init__(name="csv_tools", **kwargs)
        self.csvs = [Path(c) for c in csvs] if isinstance(csvs, list) else [Path(csvs)]
        self.register(self.get_all_data_in_csv)
        self.register(self.add_quantity_by_name)
        self.register(self.get_top3_by_quantity)
        self.register(self.get_top3_least_by_quantity)
        self.max_threshold =  max_threshold
        self.register(self.register_new_product)
        self.register(self.check_product_exist)
        self.register(self.email_send)
        self.min_threshold = min_threshold
        self.register(self.decrease_product_by_name)
        # self.register(self.email_send)
        
    def get_all_data_in_csv(self, **kwargs) -> str:
        try:
            if not self.csvs:
                return "No CSV files provided."

            all_data = {}
            for file_path in self.csvs:
                try:
                    with open(str(file_path), newline="", encoding="utf-8") as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                        csv_data = [row for row in reader]
                        all_data[Path(file_path).stem] = csv_data
                except Exception as fe:
                    logger.error(f"Error reading file {file_path}: {fe}")
                    all_data[Path(file_path).stem] = f"Error reading file: {fe}"

            return json.dumps(all_data, indent=2)
        except Exception as e:
            logger.error(f"Error reading CSVs: {e}")
            return f"Error reading CSVs: {e}"
        
    def email_send(body):
        email_sender = os.getenv("EMAIL_SENDER")
        email_receiver = os.getenv("EMAIL_RECEIVER")
        email_password = os.getenv("EMAIL_PASSWORD")

        # email_password = os.environ.get("EMAIL_PASSWORD")

        subject="Warning Mail"
        # body="This is a test email sent from Python by Kai"
        em = EmailMessage()
        em["From"] = email_sender
        em["To"] = email_receiver
        em["Subject"] = subject
        em.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())
            
    def _detect_fields(self, fieldnames: List[str]):
        quantity_field = None
        name_field = None
        for field in fieldnames:
            lower = field.lower()
            if not quantity_field and "quantity" in lower:
                quantity_field = field
            if not name_field and ("name" in lower or "product" in lower):
                name_field = field
            if quantity_field and name_field:
                break
        return quantity_field, name_field

    def get_top3_by_quantity(self, **kwargs) -> str:
        try:
            if not self.csvs:
                return "No CSV files provided."
            file_path = self.csvs[0]

            with open(str(file_path), newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                rows = list(reader)
                quantity_field, name_field = self._detect_fields(reader.fieldnames or [])

            if not quantity_field or not name_field:
                return "Required fields (name and quantity) not found."

            valid_rows = []
            for row in rows:
                try:
                    qty = int(row[quantity_field])
                    valid_rows.append((row[name_field], qty))
                except (ValueError, TypeError, KeyError):
                    continue

            if not valid_rows:
                return "No valid quantity data found."

            top3 = sorted(valid_rows, key=lambda x: x[1], reverse=True)[:3]

            result_lines = ["### Top 3 Most Stocked Items:"]
            for i, (name, qty) in enumerate(top3, 1):
                result_lines.append(f"{i}. {name} — {qty}")

            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Error getting top 3 items: {e}")
            return f"Error getting top 3 items: {e}"

    def get_top3_least_by_quantity(self, **kwargs) -> str:
        try:
            if not self.csvs:
                return "No CSV files provided."
            file_path = self.csvs[0]

            with open(str(file_path), newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
                rows = list(reader)
                quantity_field, name_field = self._detect_fields(reader.fieldnames or [])

            if not quantity_field or not name_field:
                return "Required fields (name and quantity) not found."

            valid_rows = []
            for row in rows:
                try:
                    qty = int(row[quantity_field])
                    valid_rows.append((row[name_field], qty))
                except (ValueError, TypeError, KeyError):
                    continue

            if not valid_rows:
                return "No valid quantity data found."

            least3 = sorted(valid_rows, key=lambda x: x[1])[:3]

            result_lines = ["### Top 3 Least Stocked Items:"]
            for i, (name, qty) in enumerate(least3, 1):
                result_lines.append(f"{i}. {name} — {qty}")

            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Error getting least 3 items: {e}")
            return f"Error getting least 3 items: {e}"
    
    def check_product_exist(self, name: str) -> str:
        try:
            for csv_file in self.csvs:
                path = Path(csv_file)
                if not path.exists():
                    continue  # Skip if file doesn't exist
                
                with path.open(newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Assuming product column is called 'product'
                        if row.get('name', '').strip().lower() == name.strip().lower():
                            return f"Product '{name}' found."
            
            return f"Product '{name}' not found."
        
        except Exception as e:
            return f"Error checking product: {e}"
    
    def register_new_product(self, name: str, quantity: int, descriptions: str) -> str:
        try:
            # Check if product already exists
            exists_msg = self.check_product_exist(name)
            if not "not found" in exists_msg.lower():
                return f"Product '{name}' already exists. Registration skipped."

            if not self.csvs:
                return "No CSV files configured to register the product."

            path = Path(self.csvs[0])
            file_exists = path.exists()

            fieldnames = ['id', 'name', 'quantity', 'descriptions']
            max_id = 0

            if file_exists:
                # Read existing IDs to find max
                with path.open(newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            current_id = int(row.get('id', 0))
                            if current_id > max_id:
                                max_id = current_id
                        except ValueError:
                            continue

            new_id = max_id + 1

            # Append the new product
            with path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'id': new_id,
                    'name': name,
                    'quantity': quantity,
                    'descriptions': descriptions,
                })

            return f"Product '{name}' registered successfully with id {new_id}."

        except Exception as e:
            return f"Error registering product: {e}"
 
    def decrease_product_by_name(self, name: str, amount: int) -> str:
        try:
            if not self.csvs:
                return "No CSV files configured."

            path = Path(self.csvs[0])
            if not path.exists():
                return f"CSV file {path} does not exist."

            with path.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
                if not rows:
                    return "CSV is empty."
                fieldnames = rows[0].keys()

            # Fuzzy match
            names = [row.get('name', '') for row in rows]
            match = get_close_matches(name, names, n=1, cutoff=0.6)
            if not match:
                return f"Product '{name}' not found. Try rephrasing."

            matched_name = match[0]
            product_found = False
            response = ""

            for row in rows:
                if row.get('name', '').strip().lower() == matched_name.strip().lower():
                    old_quantity = int(row.get('quantity', 0))
                    new_quantity = old_quantity - amount

                    if new_quantity < 0:
                        email_send(f"Cannot remove {amount} from '{matched_name}'. It would result in negative stock.")
                        return f"Cannot remove {amount} from '{matched_name}'. It would result in negative stock."

                    row['quantity'] = str(new_quantity)
                    product_found = True

                    response = (
                        f"Product '{matched_name}' matched.\n"
                        f"- Quantity before: {old_quantity}\n"
                        f"- Quantity after: {new_quantity}\n"
                    )

                    if new_quantity == 0:
                        response += "⚠️ Product is now out of stock.\n"
                        email_send(response)

                    break

            if not product_found:
                return f"Product '{name}' not found."

            # Write updated rows back
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            return response.strip()

        except Exception as e:
            return f"Error decreasing product: {e}"
    
    def add_quantity_by_name(self, name: str, amount: int) -> str:
        try:
            if not self.csvs:
                return "No CSV files provided."
            target_file = self.csvs[0]
            updated_rows = []
            found = False
            updated_quantity = None
            original_quantity = None
            matched_name = None
            with open(str(target_file), newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

            # Detect fields case-insensitive
            quantity_field, name_field = self._detect_fields(fieldnames)
            if not quantity_field or not name_field:
                return "CSV must contain 'name' and 'quantity' fields."

            # Fuzzy match for name (case-insensitive)
            names = [row[name_field].lower() for row in rows if row.get(name_field)]
            close_matches = difflib.get_close_matches(name.lower(), names, n=1, cutoff=0.6)

            if not close_matches:
                return f"No matching product found for '{name}'."

            matched_name_lower = close_matches[0]

            for row in rows:
                if row.get(name_field, "").lower() == matched_name_lower:
                    try:
                        original_quantity = int(row[quantity_field])
                        updated_quantity = original_quantity + amount
                        if updated_quantity >=self.max_threshold:
                            body = f"The quantity {matched_name} has reach max threshold limit ({self.max_threshold})"
                            email_send(body)
                            return f"Email has sent to user. {body}"
                        row[quantity_field] = str(updated_quantity)
                        found = True
                        matched_name = row[name_field]
                    except ValueError:
                        return f"Invalid quantity value for product '{matched_name}'."
                updated_rows.append(row)

            if not found:
                return f"No product updated for '{name}'."

            with open(str(target_file), "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(updated_rows)

            return (
                f"Product '{matched_name}' updated. "
                f"Quantity changed from {original_quantity} to {updated_quantity}."
            )

        except Exception as e:
            logger.error(f"Error updating quantity: {e}")
            return f"Error updating quantity: {e}"

async def get_csv_agent() -> Agent:
    agentic_rag_agent: Agent = Agent(
        model=Ollama(id="qwen3:4b"),
         instructions = dedent("""\
    You are a CSV Assistant.

    When asked to summarize a CSV file:
    - Display the column names and their data types
    - Show the total number of rows and count of missing (null) values per column
    - List the top 3 most stocked items 
    - List the top 3 least stocked items

    When asked to update a product:
    - Increase the quantity of the specified product, even if the name is slightly misspelled (use approximate matching)
    - Confirm and display the updated quantity
    - Tell the user what has changed in csv
    - If the updated quantity reaches a predefined limit, notify the user that a warning mail has been sent.
    
    When asked to register a new product:
    - Add the new product with all required details (such as name, quantity, descriptions).
    - Warning user which field to register has missing.
    - Confirm successful registration by displaying the new product’s information.
    
    When asked to remove a product (get product out):
    - Decrease the quantity of the specified product using approximate matching
    - Report what changed in the CSV
    
    
"""),
        tools=[CSVCustomTools(csvs=["/home/veronrd/chatbot/excel/data/khohang.csv"])],
        markdown=True,
    )
    return agentic_rag_agent

