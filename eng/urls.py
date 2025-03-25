"""
URL configuration for aitutor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    # path('',views.practice_session,name= 'practice_session'),
    path('practice_session/',views.practice_session, name = 'practice_session'),
    path('idioms/',views.idioms, name='idioms'),
    path('sample/', views.sample, name = 'sample'),
    path('report/' , views.report, name = 'report'),
    path('random/', views.generate_words_view, name='random'), 
    path('',views.index, name = 'index'),
    path('index/', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('select_topic/', views.select_topic, name='select_topic'),
    path('talk_about_anything/', views.talk_about_anything, name='talk_about_anything'),
    path('register/', views.RegisterUser, name='register'),
    path('login/', views.LoginUser, name='login'),
    path('adlogin/', views.adlogin, name='adlogin'),
    path('adhome/', views.adhome, name='adhome'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('chatbot1/', views.chatbot1, name='chatbot1'),
    path('profile/', views.profile, name='profile'),
    path('logout/', views.logout, name='logout'),
    path('editprofile/', views.editprofile, name='editprofile'),
    path('process_voice/', views.process_voice, name='process_voice'),
    path('process_voice1/', views.process_voice1 , name ='process_voice1'),
    path('userlist/', views.userlist, name='userlist'),
    path('deleteuser/<int:id>', views.deleteuser, name='deleteuser'),
    path('feedback/', views.feedback_rate, name='feedback'),
    path('feedbacklist/', views.feedbacklist, name='feedbacklist'),
    path('generate_idioms_phrases/', views.generate_idioms_phrases, name = 'generate_idioms_phrases'),
    path('generate_words/', views.generate_words_view, name='generate_words'),
    path('record-and-transcribe/<int:word_id>/', views.record_and_transcribe_view, name='record_and_transcribe'),
    path('analyze-transcription/<int:word_id>/', views.analyze_transcription_view, name='analyze_transcription'),
    path('analyze_grammar/', views.analyze_grammar_view, name='analyze_grammar'),
    path('process_voice/<str:topic>/', views.process_voice_dynamic, name='process_voice_dynamic'),
]
