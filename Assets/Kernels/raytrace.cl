#define MAX_RENDER_DIST 1000000.0f
#define EPSILON 0.001f
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

struct RayHit InterScene(struct Ray* ray, global float* arr, const uint arrlen){
    struct RayHit closest = NullRayHit();
    for(int i = 0; i < arrlen; i++){
        int off = i * 4;
        float3 spos = (float3)(arr[off + 0], arr[off + 1], arr[off + 2]);
        float srad = arr[off + 3];
        struct RayHit hit = InterSphere(ray, spos, srad);
        if(closest.t > hit.t)
            closest = hit;
    }
    return closest;
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
    __global float *sc_spheres,
    const uint sc_spheres_count,
    __global float *sc_lights,
    const uint sc_lights_count
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
    //intersect all spheres
    struct RayHit closest = InterScene(&ray, sc_spheres, sc_spheres_count);
    float col = 0.0f;
    if(col >= MAX_RENDER_DIST) col = -1.0f;
    else{
        for(int i = 0; i < sc_lights_count; i++){
            int off = i * 4;
            float3 lpos = (float3)(sc_lights[off + 0], sc_lights[off + 1], sc_lights[off + 2]);
            float lpow = sc_lights[off + 3];
            struct Ray lray;
            float3 toL = normalize(lpos - closest.pos);
            lray.pos = closest.pos + toL * EPSILON;
            lray.dir = toL;
            struct RayHit lhit = InterScene(&ray, sc_spheres, sc_spheres_count);
            float isLit = 1.0f;
            if(lhit.t >= MAX_RENDER_DIST)
                isLit = 0.0f;
            col += isLit * lpow * max(AMBIENT, dot(closest.nor, toL));
        }
    }
    r = max(0.0f, col) * 100;
    //combine rgb for final colour
    int fres = (r << 16) + (g << 8) + b;

#ifdef GLINTEROP
    int2 pos = (int2)(x, y);
	write_imagef(image_buffer, pos, fres);
#else
    image_buffer[pixid] = fres;
#endif
}