from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    Invoice: str
    Quantity: int
    InvoiceDate: str
    Price: float
    # The Python variable is 'customer_id', but it listens for 'Customer ID'
    customer_id: int = Field(alias="Customer ID")