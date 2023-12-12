from fastapi import FastAPI, Request,  Form
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import json
from utils import recommendation_functions 
from fastapi.staticfiles import StaticFiles
from loguru import logger

 
app = FastAPI()


@app.get("/")
async def main():
    return RedirectResponse(url="/home")

logger.info("Initialized app")
app.mount("/static", StaticFiles(directory="static"), name="static")

with open("movies.json", "r") as f:
    movie_data = json.load(f)


movie_titles = [movie["title"] for movie in movie_data]
movie_titles = sorted(movie_titles, key=lambda x: "".join([i for i in x if i.isalpha()]).lower())
movie_titles = sorted(movie_titles, key=lambda x: x[0].isnumeric())



templates = Jinja2Templates(directory="templates")

@app.get("/home")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/recommendation/{option}")
@app.post("/recommendation/{option}")
async def get_recommendation(request: Request, option: int):
    if option == 1:
        return templates.TemplateResponse("movie_selection.html", {"request": request, "movies": movie_titles})
    if option == 2:
        return templates.TemplateResponse("movie_selection2.html", {"request": request, "movies": movie_titles})
    if option == 3:
        return templates.TemplateResponse("movie_selection3.html", {"request": request, "movies": movie_titles})


@app.post("/recommendations")
async def post_recommendation(request: Request):
    form = await request.form()
    selected_movie1, selected_movie2, selected_movie3, selected_movie4, selected_movie5 = None, None, None, None, None
    get_recommendations1, get_recommendations2, get_recommendations3 = [], [], []
    selected_movie1, selected_movie2, selected_movie3, selected_movie4, selected_movie5 = form.get("movie1"), form.get("movie2"), form.get("movie3"), form.get("movie4"), form.get("movie5")
    logger.info(f"selected_movie1: {selected_movie1}")
    logger.info(f"selected_movie2: {selected_movie2}")
    logger.info(f"selected_movie3: {selected_movie3}")
    logger.info(f"selected_movie4: {selected_movie4}")
    logger.info(f"selected_movie5: {selected_movie5}")
    
    movie_names = [selected_movie1, selected_movie2, selected_movie3, selected_movie4, selected_movie5]
    movie_names = [x for x in movie_names if x is not None]
    movie_names = list(dict.fromkeys(movie_names))
    movies_len = len(movie_names)
    logger.info(f"movies_len: {movies_len}")
    
    if movies_len == 1:
        logger.info("Hybrid for one user")
        get_recommendations1 = recommendation_functions.get_hybrid_recommendations(selected_movie1)
        recommendations_movie = get_recommendations1
        selected_movie = selected_movie1
        
    elif movies_len == 2:
        logger.info("Hybrid for two users")
        get_recommendations2 = recommendation_functions.get_hybrid_recommendations_for_two_users(selected_movie1, selected_movie2)
        recommendations_movie = get_recommendations2
        selected_movie = f"{selected_movie1} and {selected_movie2}"
    elif movies_len > 2:
        logger.info("Knn for group")
        get_recommendations3 = recommendation_functions.get_recommendations_for_multiple_users(movie_names)
        recommendations_movie = get_recommendations3
        selected_movie = ", ".join(movie_names)
    else:
        return "Please select at least one movie from available options."
        
    return templates.TemplateResponse("show_recommendations.html", {
        "request": request, 
        "selected_movie": selected_movie,
        "recommendations_movie": recommendations_movie,
    })


