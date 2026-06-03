#We start by setting up a connection to the MySQL database using mysql.connector
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import mysql.connector
import hashlib

# Connect to the database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="my_internship"
)

# Create a cursor object
cursor = mydb.cursor()
#Define a Pydantic model for our database schema.
app = FastAPI()
class DBModel(BaseModel):
    username: str
    password: str
    email: str

#1Select All Users
@app.get("/users", status_code=status.HTTP_302_FOUND)
def select_users():
    select_query = "SELECT * FROM users"
    cursor.execute(select_query)
    results = cursor.fetchall()
    return results

#2Get User By ID
@app.get("/users/{user_id}", status_code=status.HTTP_200_OK)
def get_user_by_id(user_id: int):
    select_query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(select_query, (user_id,))
    result = cursor.fetchone()
    if result:
        return result
    else:
        raise HTTPException(status_code=404, detail="User not found")

#3. Insert a New User
@app.post("/users", status_code=status.HTTP_201_CREATED)
def insert_user(user: DBModel):
    # Hash the password using SHA-256
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()

    insert_query = """
    INSERT INTO users (username, password, email)
    VALUES (%s, %s, %s)
    """
    values = (user.username, hashed_password, user.email)

    try:
        cursor.execute(insert_query, values)
        mydb.commit()
    except mysql.connector.Error as err:
        raise HTTPException(status_code=400, detail=f"Error: {err}")

    return {"message": "User inserted successfully"}

#4. Update a User
@app.put("/users/{user_id}", status_code=status.HTTP_200_OK)
def update_user(user_id: int, user: DBModel):
    # Hash the password using SHA-256
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()

    update_query = """
    UPDATE users
    SET username = %s, password = %s, email = %s
    WHERE id = %s
    """
    values = (user.username, hashed_password, user.email, user_id)

    cursor.execute(update_query, values)
    mydb.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated successfully"}
    
    #5. Delete a User
@app.delete("/users/{user_id}", status_code=status.HTTP_200_OK)
def delete_user(user_id: int):
    delete_query = "DELETE FROM users WHERE id = %s"
    cursor.execute(delete_query, (user_id,))
    mydb.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

#successfully completed