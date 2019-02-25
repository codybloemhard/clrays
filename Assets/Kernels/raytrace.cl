#define MAX_RENDER_DIST 1000000.0f
#define EPSILON 0.001f
#define PI4 12.5663f
#define AMBIENT 0.05f

struct RayHit{
    float3 pos;
    float3 nor;
    float t;
};

struct RayHit NullRayHit(){
    struct RayHit hit;
    hit.t = MAX_RENDER_DIST;
    return hit;
}

struct Ray{
    float3 pos;
    float3 dir;
};

#define SC_LIGHT 0
#define SC_SPHERE 1
#define SC_PLANE 2

struct Scene{
    global float* items;
    global int* params;
};

global int ScGetStart(int type, struct Scene *scene){
    return scene->params[type * 3 + 2];
}

global int ScGetCount(int type, struct Scene *scene){
    return scene->params[type * 3 + 1];
}

global int ScGetStride(int type, struct Scene *scene){
    return scene->params[type * 3 + 0];
}

float dist2(float3 a, float3 b){
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z);
}

struct RayHit InterSphere(struct Ray* r, float3 spos, float srad){
    float3 l = spos - r->pos;
    float tca = dot(r->dir, l);
    float d = tca*tca - dot(l, l) + srad*srad;
    if(d < 0) return NullRayHit();
    float t = tca - sqrt(d);
    if(t < 0){
        t = tca + sqrt(d);
        if(t < 0) return NullRayHit();
    }
    struct RayHit hit;
    hit.t = t;
    hit.pos = r->pos + r->dir * t;
    hit.nor = (hit.pos - spos) / srad;
    return hit;
}

struct RayHit InterPlane(struct Ray* r, float3 ppos, float3 pnor){
    float divisor = dot(r->dir, pnor);
    if(fabs(divisor) < EPSILON) return NullRayHit();
    float3 planevec = ppos - r->pos;
    float t = dot(planevec, pnor) / divisor;
    if(t < EPSILON) return NullRayHit();
    struct RayHit hit;
    hit.t = t;
    hit.pos = r->pos + r->dir * t;
    hit.nor = pnor;
    return hit;
}

void InterSpheres(struct RayHit *closest, struct Ray *ray, global float *arr, const int count, const int start, const int stride){
    for(int i = 0; i < count; i++){
        int off = start + i * stride;
        float3 spos = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
        float srad = arr[off + 3];
        struct RayHit hit = InterSphere(ray, spos, srad);
        if(closest->t > hit.t)
            *closest = hit;
    }
}

void InterPlanes(struct RayHit *closest, struct Ray *ray, global float *arr, const int count, const int start, const int stride){
    for(int i = 0; i < count; i++){
        int off = start + i * stride;
        float3 ppos = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
        float3 pnor = (float3)(arr[off + 3], arr[off + 4], arr[off + 5]);
        struct RayHit hit = InterPlane(ray, ppos, pnor);
        if(closest->t > hit.t)
            *closest = hit;
    }
}

struct RayHit InterScene(struct Ray *ray, struct Scene *scene){
    struct RayHit closest = NullRayHit();
    InterSpheres(&closest, ray, scene->items, ScGetCount(SC_SPHERE, scene), ScGetStart(SC_SPHERE, scene), ScGetStride(SC_SPHERE, scene));
    InterPlanes(&closest, ray, scene->items, ScGetCount(SC_PLANE, scene), ScGetStart(SC_PLANE, scene), ScGetStride(SC_PLANE, scene));
    return closest;
}
//get diffuse light for hit for a light
float DiffuseSingle(float3 lpos, float lpow, struct RayHit *hit, struct Scene *scene){
    float3 toL = normalize(lpos - hit->pos);
    float angle = dot(hit->nor, toL);
    if(angle <= EPSILON)
        return 0.0f;
    float d2 = dist2(hit->pos, lpos);
    float power = lpow / (PI4 * d2);
    if(power < 0.01f)
        return 0.0f;
    struct Ray lray;
    lray.pos = hit->pos + toL * EPSILON;
    lray.dir = toL;
    struct RayHit lhit = InterScene(&lray, scene);
    float isLit = 1.0f;
    if(lhit.t * lhit.t <= d2)
        isLit = 0.0f;
    return isLit * power * max(0.0f, angle);
}
//get diffuse light of hit with all lights
float Diffuse(struct RayHit *hit, struct Scene *scene){
    float col = 0.0f;
    global float* arr = scene->items;
    int count = ScGetCount(SC_LIGHT, scene);
    int stride = ScGetStride(SC_LIGHT, scene);
    int start = ScGetStart(SC_LIGHT, scene);
    for(int i = 0; i < count; i++){
        int off = start + i * stride;
        float3 lpos = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
        float lpow = arr[off + 3];
        col += DiffuseSingle(lpos, lpow, hit, scene);
    }
    return max(col, AMBIENT);
}

#ifdef GLINTEROP
__kernel void render(
    write_only image2d_t image_buffer,
#else
__kernel void render(
    __global int *image_buffer,
#endif
    const uint w,
    const uint h,
    __global int *sc_params,
    __global float *sc_items
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pixid = x + y * w;
    //(0,0) is in middle of screen
    float2 uv = (float2)(((float)x / w) - 0.5f, ((float)y / h) - 0.5f);
    uv *= (float2)((float)w/h, -1.0f);
    //colours
    char r = 0;
    char g = 0;
    char b = 0;
    //construct ray, simple perspective
    struct Ray ray;
    ray.pos = (float3)(0,0,0);
    ray.dir = normalize((float3)(uv.x,uv.y,-1) - ray.pos);
    //Scene
    struct Scene scene;
    scene.params = sc_params;
    scene.items = sc_items;

    struct RayHit closest = InterScene(&ray, &scene);
    float col = 0.0f;
    if(closest.t >= MAX_RENDER_DIST) col = -1.0f;
    else col = Diffuse(&closest, &scene);
    r = clamp(col, 0.0f, 1.0f) * 255;
    //combine rgb for final colour
    int fres = (r << 16) + (g << 8) + b;

#ifdef GLINTEROP
    int2 pos = (int2)(x, y);
	write_imagef(image_buffer, pos, fres);
#else
    image_buffer[pixid] = fres;
#endif
}