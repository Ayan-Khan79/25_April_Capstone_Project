# Mock CRM system
def get_customer_data(user_id="123"):
    return {
        "user_id": user_id,
        "name": "John Doe",
        "order_status": "Shipped"
    }

# Mock Ticketing system
def create_ticket(issue):
    return {
        "ticket_id": "TCKT1001",
        "status": "Created",
        "issue": issue
    }