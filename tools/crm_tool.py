import pandas as pd
from langchain.tools import Tool
from typing import Optional

class CRMTool:
    def __init__(self, crm_data_path: str = "data/mock_crm_data.csv"):
        self.crm_data_path = crm_data_path
        self.df = pd.read_csv(crm_data_path)
    
    def get_customer_info(self, customer_id: str) -> str:
        """Get customer information by customer ID."""
        try:
            customer = self.df[self.df['customer_id'] == customer_id]
            if customer.empty:
                return f"Customer {customer_id} not found."
            
            info = customer.iloc[0]
            return f"""Customer Information:
- ID: {info['customer_id']}
- Name: {info['name']}
- Email: {info['email']}
- Account Status: {info['account_status']}
- Last Purchase: {info['last_purchase_date']}
- Subscription Tier: {info['subscription_tier']}"""
        except Exception as e:
            return f"Error retrieving customer information: {str(e)}"
    
    def get_customer_by_email(self, email: str) -> str:
        """Get customer information by email address."""
        try:
            customer = self.df[self.df['email'] == email]
            if customer.empty:
                return f"Customer with email {email} not found."
            
            info = customer.iloc[0]
            return f"""Customer Information:
- ID: {info['customer_id']}
- Name: {info['name']}
- Email: {info['email']}
- Account Status: {info['account_status']}
- Last Purchase: {info['last_purchase_date']}
- Subscription Tier: {info['subscription_tier']}"""
        except Exception as e:
            return f"Error retrieving customer information: {str(e)}"
    
    def get_account_status_tool(self) -> Tool:
        """Return tool for checking customer account status."""
        return Tool(
            name="get_customer_account_status",
            description="Get customer account information including status, subscription tier, and last purchase date. Use customer ID (format: CUST001) or email address.",
            func=self._handle_customer_lookup
        )
    
    def _handle_customer_lookup(self, identifier: str) -> str:
        """Handle customer lookup by ID or email."""
        if identifier.startswith("CUST"):
            return self.get_customer_info(identifier)
        elif "@" in identifier:
            return self.get_customer_by_email(identifier)
        else:
            return "Please provide a valid customer ID (e.g., CUST001) or email address."