from django.shortcuts import render
from . import logic # de la misma carpeta solver
import numpy as np
# Create your views here.

def solver_view(request):
    context = {} #envia datos a la plantilla
    if request.method == 'POST':
        results = logic.solve_graphical_method(request.POST)
        context ['results'] = results

    return render(request, 'solver/solver_page.html', context)
