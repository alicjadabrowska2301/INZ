from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
from utils import recommendation_functions

app = FastAPI()

# Load the list of movies from the JSON file
with open("movies.json", "r") as f:
    movie_data = json.load(f)

# Extract only the movie titles from the JSON data
movie_titles = [movie["title"] for movie in movie_data]

# Create an instance of the Jinja2Templates class
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Serve the HTML page with the dropdown of movies directly
@app.get("/recommendation/{option}")
@app.post("/recommendation/{option}")
async def get_recommendation(request: Request, option: int):
    if option == 1:
        return templates.TemplateResponse("movie_selection.html", {"request": request, "movies": movie_titles})
    if option == 2:
        return templates.TemplateResponse("movie_selection2.html", {"request": request, "movies": movie_titles})
    if option == 3:
        return templates.TemplateResponse("movie_selection.html", {"request": request, "movies": movie_titles})


# Handle movie selection and recommendation
@app.post("/recommendation")
async def post_recommendation(request: Request):
    form = await request.form()
    selected_movie1 = form.get("movie1")
    selected_movie2 = form.get("movie2")

    get_recommendations1 = []
    
    get_recommendations2 = []

    if selected_movie1:
        get_recommendations1 = recommendation_functions.get_hybrid_recommendations(selected_movie1)
        
        recommendations_movie = get_recommendations1
        selected_movie = selected_movie1
        
    if selected_movie2:
        get_recommendations2 = recommendation_functions.get_CB_recommendations_for_two_users(selected_movie1, selected_movie2)
        recommendations_movie = get_recommendations2
        selected_movie = f"{selected_movie} and {selected_movie2}"
    return templates.TemplateResponse("show_recommendations.html", {
        "request": request, 
        "selected_movie": selected_movie,
        "recommendations_movie": recommendations_movie,
    })
