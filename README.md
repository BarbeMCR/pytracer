# Pytracer
A Python reimplementation of the C++ raytracer from "Raytracing in a weekend".

This project was made using the Pygame library.

![Example scene](/render_final.png)

This is an example scene rendered with Pytracer.

To generate a scene of random spheres, you can simply call Pytracer this way:

`python pytracer.py output width height samples depth speed`

- *output* is the path to the output image. The image will only be rendered to the Pygame window and not saved if this is omitted.
- *width* is the width of the output image in pixels. It is 1280 if omitted.
- *height* is the height of the output image in pixels. It is 720 if omitted.
- *samples* is the number of samples per pixels used in antialiasing. It is 500 if omitted.
- *depth* is the maximum number of bounces a ray can do. It is 50 if omitted. You can't set this higher than 1000.
- For an explanation of *speed*, call `python pytracer.py help`.

Calling `python pytracer.py help` will explain all of the arguments.
