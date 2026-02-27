from django.urls import path
from benchmark import views

app_name = "benchmark"

urlpatterns = [
    path("", views.home, name="home"),
    path("run/", views.run_benchmark, name="run"),
    path("results/<uuid:session_id>/", views.results, name="results"),
    path("history/", views.history, name="history"),
]
