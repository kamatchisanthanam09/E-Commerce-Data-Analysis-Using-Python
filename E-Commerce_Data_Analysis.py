import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#DATA SET
np.random.seed(42)

products = {
    "Laptop": ("Electronics", 50000, 40000),
    "Mobile": ("Electronics", 20000, 15000),
    "Headphones": ("Electronics", 2000, 1200),
    "T-Shirt": ("Fashion", 800, 400),
    "Jeans": ("Fashion", 1500, 900),
    "Shoes": ("Fashion", 2500, 1500),
    "Mixer": ("Home Appliances", 3500, 2500),
    "Fan": ("Home Appliances", 2200, 1600),
    "Books": ("Books", 500, 250),
    "Notebook": ("Books", 100, 50)
}

product_names = list(products.keys())

data = []

for i in range(1, 1001):

    product = np.random.choice(product_names)

    category, price, cost = products[product]

    quantity = np.random.randint(1, 6)

    discount = np.random.choice([0, 5, 10, 15, 20])

    sales = quantity * price * (1 - discount / 100)

    profit = sales - (quantity * cost)

    data.append([
        i,
        pd.Timestamp("2024-01-01")
        + pd.Timedelta(days=int(np.random.randint(0, 731))),
        np.random.randint(101, 301),
        product,
        category,
        np.random.choice(
            ["North", "South", "East", "West"]
        ),
        quantity,
        price,
        discount,
        np.random.choice(
            ["UPI", "Credit Card", "Debit Card", "COD"]
        ),
        cost,
        sales,
        profit
    ])

columns = [
    "Order_ID",
    "Order_Date",
    "Customer_ID",
    "Product",
    "Category",
    "Region",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Payment_Method",
    "Cost_Price",
    "Total_Sales",
    "Profit"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("ecommerce_data.csv", index=False)

print("Dataset Created Successfully!")

df = pd.read_csv("ecommerce_data.csv")

print(df.head())

#=========================
#Data Understanding
#========================

#1.Display First 10 Rows
print(df.head(10))

#2.Display Last 5 Rows
print(df.tail())

#3.Find Dataset Shape
print(df.shape)

#4.Display Column Names
print(df.columns)

#5.Dataset Information
print(df.info())

#6.Statistical Summary
print(df.describe())

#7.Check Data Types
print(df.dtypes)

#==========================
#Data Cleaning
#==========================

#8.Check Missing Values
print(df.isnull().sum())

#9.Handle Missing Values
df = df.dropna()
print(df)

#10.Check Duplicate Records
print("Checking duplicates")
print(df.duplicated().sum())
print("Duplicate checking completed")

#11.Convert Date Column
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
print(df.dtypes)

#12.Check Invalid Values
print("Invalid Quantity")
print((df["Quantity"] <= 0).sum())
print("Invalid Unit_Price ")
print((df["Unit_Price"] <= 0).sum())
print("Invalid Discount")
print(((df["Discount"] < 0) |
       (df["Discount"] > 100)).sum())

#13.Create New Columns
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.month_name()
print(df.head())

#============================
#Exploratory Data Analysis
#============================

#14.Total Sales Revenue
total_sales = df["Total_Sales"].sum()
print("Total Sales Revenue: ₹", round(total_sales, 2))

#15.Total Orders
total_orders = df["Order_ID"].nunique()
print("Total Orders:", total_orders)

#16.Total Customers
total_customers = df["Customer_ID"].nunique()
print("Total Customers:", total_customers)

#17.Total Quantity Sold
total_quantity = df["Quantity"].sum()
print("Total Quantity Sold:", total_quantity)

#18.Average Order Value
df["Total_Amount"] = (df["Quantity"] * df["Unit_Price"]* (1 - df["Discount"] / 100))
average_order_value = df["Total_Amount"].mean()
print("Average Order Value: ₹", round(average_order_value, 2))

#========================
#Sales Analysis
#========================

#19.Category-wise Sales
category_sales = (
   df.groupby("Category")["Total_Amount"].sum().sort_values(ascending=False))
print(category_sales)

#.Visualization (Category-wise Sales)
category_sales.plot(
    kind="bar",
    figsize=(8, 5))

plt.title("Category-wise Sales")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#20.Region-wise Sales
region_sales = (
    df.groupby("Region")["Total_Sales"].sum().sort_values(ascending=False))
print(region_sales)

#Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    x=region_sales.index, y=region_sales.values)

plt.title("Region-wise Sales")
plt.xlabel("Region")
plt.ylabel("Revenue")
plt.show()


#21.Product-wise Sales
product_sales = (
    df.groupby("Product")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)
print("\nProduct-wise Sales:")
print(product_sales)

# Visualization
plt.figure(figsize=(10, 5))
sns.barplot(
    x=product_sales.index,
    y=product_sales.values,
    color="skyblue")
plt.title("Product-wise Revenue")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#22.Top 5 Selling Products
top_products = (
    df.groupby("Product")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(5))
print(top_products)

#Visualization
top_products.plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Top 5 Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")

plt.tight_layout()
plt.show()

#23.Top 10 Customers
top_customers = (
    df.groupby("Customer_ID")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customers)

#Visualization
top_customers.plot(
    kind="bar",
    figsize=(10, 5))
plt.title("Top 10 Customers by Revenue")
plt.xlabel("Customer ID")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

#===================================
#  Monthly Sales Analysis
#===================================
#24.Monthly Sales Trends
monthly_sales = (
    df.groupby(
        df["Order_Date"].dt.to_period("M")
    )["Total_Sales"]
    .sum()
)
print(monthly_sales)

#25.Category-wise Profit
category_profit = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(category_profit)

#Visualization
category_profit.plot(
    kind="bar",
    figsize=(8, 5)
)
plt.title("Category-wise Profit")
plt.xlabel("Category")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#26.Region-wise Profit
region_profit = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
print(region_profit)

#27.Profit Margin
total_profit = df["Profit"].sum()
profit_margin = (
    total_profit / total_sales
) * 100
print(
    "Profit Margin:",
    round(profit_margin, 2),
    "%"
)

#==============================
#Discount Analysis
#==============================

#28.Discount-wise Sales
discount_sales = (
    df.groupby("Discount")["Total_Sales"]
    .sum()
)
print(discount_sales)

#Visualization
sns.barplot(
    x=discount_sales.index,
    y=discount_sales.values
)
plt.title("Discount-wise Sales")
plt.xlabel("Discount Percentage")
plt.ylabel("Revenue")
plt.show()

#==============================
#Customer Behaviour Analysis
#==============================

#29.Customer Purchase Frequency
customer_frequency = (
    df.groupby("Customer_ID")["Order_ID"]
    .nunique()
    .sort_values(ascending=False)
)
print(customer_frequency.head(10))

#30.Repeat Customers
repeat_customers = (
    customer_frequency > 1
).sum()
print("Repeat Customers:", repeat_customers)

#31.Customer Lifetime Revenue
customer_revenue = (
    df.groupby("Customer_ID")["Total_Sales"]
    .sum()
    .sort_values(ascending=False)
)
print(customer_revenue.head(10))

#=================================
#Additional Data Visualizations
#==================================

#32.Sales Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df["Total_Sales"],
    bins=30,
    kde=True
)
plt.title("Sales Distribution")
plt.xlabel("Sales Amount")
plt.show()

#33.Correlation Heatmap
numeric_columns = [
    "Quantity",
    "Unit_Price",
    "Discount",
    "Cost_Price",
    "Total_Sales",
    "Profit"
]
correlation = df[numeric_columns].corr()
plt.figure(figsize=(9, 6))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

#34.Sales by Year
yearly_sales = (
    df.groupby("Year")["Total_Sales"]
    .sum()
)
yearly_sales.plot(
    kind="bar",
    figsize=(7, 5)
)
plt.title("Yearly Sales")
plt.xlabel("Year")
plt.ylabel("Revenue")
plt.show()

