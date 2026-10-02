"""Flask Todo Application - A simple CRUD task manager."""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from database import db, Todo

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///todos.db"
db.init_app(app)

@app.route("/")
def index():
    """Display all todo items sorted by creation date."""
    todos = Todo.query.order_by(Todo.created_at.desc()).all()
    return render_template("index.html", todos=todos)

@app.route("/add", methods=["POST"])
def add_todo():
    """Create a new todo item from form data."""
    title = request.form.get("title", "").strip()
    if title:
        todo = Todo(title=title, priority=request.form.get("priority", "medium"))
        db.session.add(todo)
        db.session.commit()
    return redirect(url_for("index"))

@app.route("/toggle/<int:todo_id>")
def toggle_todo(todo_id):
    """Toggle the completion status of a todo item."""
    todo = Todo.query.get_or_404(todo_id)
    todo.completed = not todo.completed
    db.session.commit()
    return redirect(url_for("index"))

@app.route("/delete/<int:todo_id>")
def delete_todo(todo_id):
    """Delete a todo item by its ID."""
    todo = Todo.query.get_or_404(todo_id)
    db.session.delete(todo)
    db.session.commit()
    return redirect(url_for("index"))

@app.route("/api/todos")
def api_list_todos():
    """REST API endpoint to list all todos as JSON."""
    todos = Todo.query.all()
    return jsonify([t.to_dict() for t in todos])

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
