# Pytracer rev. 1
# Translated into Python from the original C++ code in "Ray Tracing In One Weekend"
# For the Python re-implementation (Pytracer): Copyright (c) 2023  BarbeMCR

import pygame
import sys
import math
import random
import ctypes

if len(sys.argv)==2 and sys.argv[1].lower()=='help':
    print()
    print("Pytracer")
    print("Syntax:")
    print("python pytracer.py [output] [width] [height] [samples] [depth] [speed]")
    print("python pytracer.py help")
    print()
    print("Arguments:")
    print("All arguments are optional.")
    print("<path> help  -- display this help screen")
    print("output       -- output image path")
    print("width/height -- output resolution")
    print("samples      -- number of samples per pixel")
    print("depth        -- maximum bounces per ray")
    print("speed        -- can be 'fast', 'slow' or 'slower'")
    print("'fast' means that the image will be drawn to the screen only after it finished rendering")
    print("'slow' means that the image will be drawn to the screen after each scanline finished rendering")
    print("'slower' means that the image will be drawn to the screen after each pixel gets rendered")
    sys.exit()

pygame.init()
ctypes.windll.shcore.SetProcessDpiAwareness(2)
width = int(sys.argv[2]) if len(sys.argv)>2 else 1280
height = int(sys.argv[3]) if len(sys.argv)>3 else 720
aspect_ratio = width / height
samples = int(sys.argv[4]) if len(sys.argv)>4 else 500
depth = int(sys.argv[5]) if len(sys.argv)>5 else 50
screen = pygame.display.set_mode((width, height), pygame.SCALED)
render_surface = pygame.Surface((width, height))
render_start = pygame.time.get_ticks()

def get_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

def get_eta(width, height, row, col, scanline_start, pixel_start):
    """Gets the ETA.
    width -- the image width
    height -- the image height
    row -- the current row
    col -- the current column
    scanline_start -- the time at the beginning of the scanline
    pixel_start -- the time at the beginning of the pixel rendering
    """
    now = pygame.time.get_ticks()
    pixel_time = now - pixel_start
    scanline_time = now - scanline_start
    average_pixel_time = scanline_time / (col+1)
    average_time = (pixel_time+average_pixel_time*3) / 4
    eta = (average_time*width*(height-row-1) + average_time*(width-col-1)) / 1000
    eta_days = int(eta // (24*3600))
    eta %= 24*3600
    eta_hours = int(eta // 3600)
    eta %= 3600
    eta_minutes = int(eta // 60)
    eta %= 60
    eta_seconds = eta
    return eta_days, eta_hours, eta_minutes, eta_seconds

def degrees_to_radians(degrees):
    return degrees * math.pi / 180

def get_random(min, max):
    return min + (max-min)*random.random()

def random_Vector3():
    return pygame.Vector3(random.random(), random.random(), random.random())

def get_random_Vector3(min, max):
    return pygame.Vector3(get_random(min, max), get_random(min, max), get_random(min, max))

def random_in_unit_sphere():
    while True:
        p = get_random_Vector3(-1, 1)
        if p.length_squared() >= 1:
            continue
        return p

def random_in_unit_disk():
    while True:
        p = pygame.Vector3(get_random(-1, 1), get_random(-1, 1), 0)
        if p.length_squared() >= 1:
            continue
        return p

def random_unit_vector():
    return random_in_unit_sphere().normalize()

def check_Vector3_near_zero(vec):
    s = 1e-8
    return (math.fabs(vec.x)<s) and (math.fabs(vec.y)<s) and (math.fabs(vec.z)<s)

def refract(uv, n, etai_over_etat):
    cos_theta = min(-uv*n, 1)
    r_out_perp = etai_over_etat * (uv+cos_theta*n)
    r_out_parallel = -math.sqrt(math.fabs(1-r_out_perp.length_squared())) * n
    return r_out_perp + r_out_parallel

class Ray:
    def __init__(self, origin, direction):
        self.origin = pygame.Vector3(origin)
        self.direction = pygame.Vector3(direction)
    def at(self, t):
        return self.origin + t*self.direction
    def copy(self, ray):
        self.origin = ray.origin
        self.direction = ray.direction

class Hit:
    def __init__(self):
        self.p = pygame.Vector3(0, 0, 0)
        self.n = pygame.Vector3(0, 0, 0)
        self.mat_ptr = Material()
        self.t = 0
        self.front_face = False
    def copy(self, rec):
        self.p = rec.p
        self.n = rec.n
        self.mat_ptr = rec.mat_ptr
        self.t = rec.t
        self.front_face = rec.front_face
    def set_face_normal(self, ray, outward_n):
        self.front_face = ray.direction * outward_n < 0
        self.n = outward_n if self.front_face else -outward_n

class Hittable:
    def hit(self, ray, t_min, t_max, rec):
        pass

class HittableList(Hittable):
    def __init__(self, obj):
        self.objects = []
        if obj is not None:
            self.add(obj)
    def add(self, obj):
        self.objects.append(obj)
    def clear(self):
        self.objects.clear()
    def hit(self, ray, t_min, t_max, rec):
        temp_rec = Hit()
        hit_anything = False
        closest_so_far = t_max
        for obj in self.objects:
            if obj.hit(ray, t_min, closest_so_far, temp_rec):
                hit_anything = True
                closest_so_far = temp_rec.t
                rec.copy(temp_rec)
        return hit_anything

def get_ray_color(ray, world, depth):
    rec = Hit()
    if depth <= 0:
        return pygame.Vector3(0, 0, 0)
    if world.hit(ray, 0.001, math.inf, rec):
        scattered = Ray(pygame.Vector3(0, 0, 0), pygame.Vector3(0, 0, 0))
        attenuation = pygame.Vector3(0, 0, 0)
        if rec.mat_ptr.scatter(ray, rec, attenuation, scattered):
            c = get_ray_color(scattered, world, depth-1)
            if isinstance(c, float):
                x = attenuation.x * c
                y = attenuation.y * c
                z = attenuation.z * c
                return pygame.Vector3(x, y, z)
            elif isinstance(c, pygame.Vector3):
                x = attenuation.x * c.x
                y = attenuation.y * c.y
                z = attenuation.z * c.z
                return pygame.Vector3(x, y, z)
            else:  # Fallback
                return pygame.Vector(attenuation*c)
        return pygame.Vector3(0, 0, 0)
    unit_direction = ray.direction.normalize()
    t = 0.5 * (unit_direction.y+1)
    a = (1-t) * pygame.Vector3(1, 1, 1)
    b = t * pygame.Vector3(0.5, 0.7, 1)
    color = a + b
    return color

def get_color(r, g, b, samples):
    scale = 1 / samples
    r = math.sqrt(scale * r)
    g = math.sqrt(scale * g)
    b = math.sqrt(scale * b)
    r = 256 * pygame.math.clamp(r, 0, 0.999)
    g = 256 * pygame.math.clamp(g, 0, 0.999)
    b = 256 * pygame.math.clamp(b, 0, 0.999)
    return pygame.Color(int(r), int(g), int(b))

class Sphere(Hittable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.mat_ptr = material
    def hit(self, ray, t_min, t_max, rec):
        oc = ray.origin - self.center
        a = ray.direction.length_squared()
        b = oc * ray.direction
        c = oc.length_squared() - self.radius**2
        d = b**2 - a*c
        if d < 0:
            return False
        sqrtd = math.sqrt(d)
        root = (-b - sqrtd) / a
        if root < t_min or root > t_max:
            root = (-b + sqrtd) / a
            if root < t_min or root > t_max:
                return False
        rec.t = root
        rec.p = ray.at(rec.t)
        outward_n = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_n)
        rec.mat_ptr = self.mat_ptr
        return True

class Material:
    def scatter(self, ray, rec, attenuation, scattered):
        pass

class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo
    def scatter(self, ray, rec, attenuation, scattered):
        scatter_direction = rec.n + random_unit_vector()
        if check_Vector3_near_zero(scatter_direction):
            scatter_direction = rec.n
        scattered.copy(Ray(rec.p, scatter_direction))
        attenuation.update(*self.albedo)
        return True

class Metal(Material):
    def __init__(self, albedo, fuzziness):
        self.albedo = albedo
        self.fuzziness = fuzziness if fuzziness < 1 else 1
    def scatter(self, ray, rec, attenuation, scattered):
        reflected = ray.direction.normalize().reflect(rec.n)
        scattered.copy(Ray(rec.p, reflected+self.fuzziness*random_in_unit_sphere()))
        attenuation.update(*self.albedo)
        return scattered.direction*rec.n > 0

class Dielectric(Material):
    def __init__(self, index):
        self.index = index
    def scatter(self, ray, rec, attenuation, scattered):
        attenuation.update(*pygame.Vector3(1, 1, 1))
        refraction_ratio = 1/self.index if rec.front_face else self.index
        unit_direction = ray.direction.normalize()
        cos_theta = min(-unit_direction*rec.n, 1)
        sin_theta = math.sqrt(1 - cos_theta**2)
        cannot_refract = refraction_ratio*sin_theta > 1
        direction = pygame.Vector3(0, 0, 0)
        if cannot_refract or self.reflectance(cos_theta, refraction_ratio)>random.random():
            direction = unit_direction.reflect(rec.n)
        else:
            direction = refract(unit_direction, rec.n, refraction_ratio)
        scattered.copy(Ray(rec.p, direction))
        return True
    def reflectance(self, cos, ref_idx):
        r0 = (1-ref_idx) / (1+ref_idx)
        r0 **= 2
        return r0 + (1-r0)*(1-cos)**5

class Camera:
    def __init__(self, lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist):
        theta = degrees_to_radians(vfov)
        h = math.tan(theta/2)
        viewport_height = 2 * h
        viewport_width = aspect_ratio * viewport_height
        self.w = (lookfrom-lookat).normalize()
        self.u = vup.cross(self.w).normalize()
        self.v = self.w.cross(self.u)
        self.origin = lookfrom
        self.horz = focus_dist * viewport_width * self.u
        self.vert = focus_dist * viewport_height * self.v
        self.bottomleft = self.origin - self.horz/2 - self.vert/2 - focus_dist*self.w
        self.lens_radius = aperture / 2
    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u*rd.x + self.v*rd.y
        ray = Ray(self.origin+offset, self.bottomleft + s*self.horz + t*self.vert - self.origin - offset)
        return ray

def generate_random_scene():
    world = HittableList(None)
    ground_material = Lambertian(pygame.Vector3(0.5, 0.5, 0.5))
    world.add(Sphere(pygame.Vector3(0, -1000, 0), 1000, ground_material))
    for a in range(-11, 11):
        for b in range(-11, 11):
            choose_mat = random.random()
            center = pygame.Vector3(a+0.9*random.random(), 0.2, b+0.9*random.random())
            if (center - pygame.Vector3(4, 0.2, 0)).length() > 0.9:
                sphere_material = Material()  # Fallback
                if choose_mat < 0.8:
                    u = random_Vector3()
                    v = random_Vector3()
                    albedo = pygame.Vector3(u.x*v.x, u.y*v.y, u.z*v.z)
                    sphere_material = Lambertian(albedo)
                    world.add(Sphere(center, 0.2, sphere_material))
                elif choose_mat < 0.95:
                    albedo = get_random_Vector3(0.5, 1)
                    fuzz = get_random(0, 0.5)
                    sphere_material = Metal(albedo, fuzz)
                    world.add(Sphere(center, 0.2, sphere_material))
                else:
                    sphere_material = Dielectric(1.5)
                    world.add(Sphere(center, 0.2, sphere_material))
    material1 = Dielectric(1.5)
    world.add(Sphere(pygame.Vector3(0, 1, 0), 1, material1))
    material2 = Lambertian(pygame.Vector3(0.4, 0.2, 0.1))
    world.add(Sphere(pygame.Vector3(-4, 1, 0), 1, material2))
    material3 = Metal(pygame.Vector3(0.7, 0.6, 0.5), 0)
    world.add(Sphere(pygame.Vector3(4, 1, 0), 1, material3))
    return world

# World
#world = HittableList(None)
#material_center = Dielectric(1.5)
#material_center = Lambertian(pygame.Vector3(0.7, 0.3, 0.3))
#material_left = Metal(pygame.Vector3(0.8, 0.8, 0.8), 0.15)
#material_ground = Lambertian(pygame.Vector3(0.8, 0.8, 0))
#material_center = Lambertian(pygame.Vector3(0.1, 0.2, 0.5))
#material_left = Dielectric(1.5)
#material_right = Metal(pygame.Vector3(0.8, 0.6, 0.2), 0.3)
#world.add(Sphere(pygame.Vector3(0, -100.5, -1), 100, material_ground))
#world.add(Sphere(pygame.Vector3(0, 0, -1), 0.5, material_center))
#world.add(Sphere(pygame.Vector3(-1, 0, -1), 0.5, material_left))
#world.add(Sphere(pygame.Vector3(-1, 0, -1), -0.45, material_left))
#world.add(Sphere(pygame.Vector3(1, 0, -1), 0.5, material_right))
world = generate_random_scene()

# Camera
lookfrom = pygame.Vector3(13, 2, 3)
lookat = pygame.Vector3(0, 0, 0)
vup = pygame.Vector3(0, 1, 0)
#dist_to_focus = (lookfrom-lookat).length()
dist_to_focus = 10
#aperture = 2
aperture = 0.1
camera = Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus)

last_update = 0
# Renderer
for i, row in enumerate(range(height)):
    start_scanline = pygame.time.get_ticks()
    if len(sys.argv)<=6 or (len(sys.argv)>6 and (sys.argv[6].lower()!='slow' and sys.argv[6].lower()!='slower')):
        pygame.display.set_caption(f"Pytracer ({height-1-row} scanlines remaining)")
    #get_events()
    for j, col in enumerate(range(width)):
        start_pixel = pygame.time.get_ticks()
        color = pygame.Vector3(0, 0, 0)
        for sample in range(samples):
            u = (col+random.random()) / (width-1)
            v = (row+random.random()) / (height-1)
            ray = camera.get_ray(u, v)
            color += get_ray_color(ray, world, depth)
        get_events()
        render_surface.set_at((col, height-1-row), get_color(*color, samples))
        if len(sys.argv)>6 and sys.argv[6].lower()=='slower':
            screen.blit(render_surface, (0, 0))
            pygame.display.update()
        if len(sys.argv)>6 and (sys.argv[6].lower()=='slow' or sys.argv[6].lower()=='slower') and (pygame.time.get_ticks()-last_update)>=500:
            eta = get_eta(width, height, i, j, start_scanline, start_pixel)
            last_update = pygame.time.get_ticks()
            if eta[0] > 0:
                pygame.display.set_caption(f"Pytracer ({height-row} scanlines remaining, ETA: {eta[0]} days, {eta[1]} hours, {eta[2]} minutes, {eta[3].__format__('.3f')} seconds)")
            elif eta[1] > 0:
                pygame.display.set_caption(f"Pytracer ({height-row} scanlines remaining, ETA: {eta[1]} hours, {eta[2]} minutes, {eta[3].__format__('.3f')} seconds)")
            elif eta[2] > 0:
                pygame.display.set_caption(f"Pytracer ({height-row} scanlines remaining, ETA: {eta[2]} minutes, {eta[3].__format__('.3f')} seconds)")
            else:
                pygame.display.set_caption(f"Pytracer ({height-row} scanlines remaining, ETA: {eta[3].__format__('.3f')} seconds)")
    if len(sys.argv)>6 and sys.argv[6].lower()=='slow':
        screen.blit(render_surface, (0, 0))
        pygame.display.update()

# Save rendered surface to file
if len(sys.argv) > 1:
    pygame.image.save(render_surface, sys.argv[1])

# Show render time
render_time = (pygame.time.get_ticks() - render_start) / 1000
render_days = int(render_time // (24*3600))
render_time %= 24*3600
render_hours = int(render_time // 3600)
render_time %= 3600
render_minutes = int(render_time // 60)
render_time %= 60
render_seconds = render_time
if render_days > 0:
    pygame.display.set_caption(f"Raytracer (render complete in {render_days} days, {render_hours} hours, {render_minutes} minutes, {render_seconds.__format__('.3f')} seconds)")
    print(f"Completed rendering in {render_days} days, {render_hours} hours, {render_minutes} minutes, {render_seconds.__format__('.3f')} seconds.")
elif render_hours > 0:
    pygame.display.set_caption(f"Raytracer (render complete in {render_hours} hours, {render_minutes} minutes, {render_seconds.__format__('.3f')} seconds)")
    print(f"Completed rendering in {render_hours} hours, {render_minutes} minutes, {render_seconds.__format__('.3f')} seconds.")
elif render_minutes > 0:
    pygame.display.set_caption(f"Raytracer (render complete in {render_minutes} minutes, {render_seconds.__format__('.3f')} seconds)")
    print(f"Completed rendering in {render_minutes} minutes, {render_seconds.__format__('.3f')} seconds.")
else:
    pygame.display.set_caption(f"Raytracer (render complete in {render_seconds.__format__('.3f')} seconds)")
    print(f"Completed rendering in {render_seconds.__format__('.3f')} seconds.")

while True:
    get_events()
    screen.fill('black')
    screen.blit(render_surface, (0, 0))
    pygame.display.update()