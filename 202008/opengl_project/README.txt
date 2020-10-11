2020.10.09 A demo of opengl
sudo apt-get install build-essential libgl1-mesa-dev
sudo apt-get install freeglut3-dev
sudo apt-get install libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev

gcc windows_ball.c -o windows_ball -lGL -lglut
./windows_ball

2020.10.10 Learn OpenGL CN https://learnopengl-cn.github.io
Part 0
  install GLFW3 GLM
Part 1.1 GLFW
  #include <GLFW\glfw3.h>
  -lGLEW -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi
