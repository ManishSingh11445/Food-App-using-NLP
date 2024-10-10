import mysql.connector
import random
from decimal import Decimal

def create_connection():
    from decimal import Decimal
    # Connect to MySQL
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="food_app"
    )
    cursor = db_connection.cursor()
    return cursor,db_connection


# Insert single row into the products table
# insert_query = "INSERT INTO products (product_name, price) VALUES (%s, %s)"

def generate_order_id():
    return random.randint(1000, 9999)


def is_order_id_unique(order_id,cursor):
    cursor.execute("SELECT COUNT(*) FROM ordered_items WHERE order_id = %s", (order_id,))
    count = cursor.fetchone()[0]
    return count == 0

def is_p_id_unique(p_id,cursor):
    cursor.execute("SELECT COUNT(*) FROM ordered_items WHERE p_id = %s", (p_id,))
    count = cursor.fetchone()[0]
    return count == 0



def insert_data(order_items):
    cursor,db_connection = create_connection()
    final_cost = 0
    order_id = 0
    tup_items = []
    for item,quantity in order_items.items():
        tup_items.append(item)
    placeholders = ', '.join(['%s'] * len(tup_items))
    get_query = f"SELECT * from food_items where item in ({placeholders})"
    items = tuple(tup_items)
    try:
        cursor.execute(get_query, items)
        results = cursor.fetchall()
        print(results)  # Commit the transaction
        # for item in order_items:
            # order_id = generate_order_id()
        order_id = generate_order_id()
        while not is_order_id_unique(order_id, cursor):
            order_id = generate_order_id()
        for result in results:
            p_id = generate_order_id()
            while not is_p_id_unique(p_id, cursor):
                p_id = generate_order_id()
            cost = 0
            item_id, item_name, price = result
            cost = float(price) * order_items[item_name]
            final_cost += cost
            insert_query = "INSERT INTO ordered_items (p_id, order_id, item, quantity, cost) VALUES (%s, %s, %s, %s, %s)"
            data_tuple = (p_id, order_id, item_name, order_items[item_name] , cost)
            try:
                cursor.execute(insert_query, data_tuple)
            except mysql.connector.Error as err:
                print(f"Error: {err}")
        db_connection.commit()
        print("Order items inserted successfully!")
        print(f"order_id: {order_id}")
        print(f"final_cost: {final_cost}")
        return order_id, final_cost
    # print(f"Inserted {cursor.rowcount} row(s) successfully!")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

    # Close the cursor and connection
    cursor.close()
    db_connection.close()

# if __name__=='__main__':
#     # order_items = {"lassi": 2, "sandwich": 1}
#     insert_data(order_items)