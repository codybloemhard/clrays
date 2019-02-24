#define MAX_RENDER_DIST 1000000.0f

struct RayHit{
    float3 pos;
    float3 nor;
    float t;
};

struct Ray{
    float3 pos;
    float3 dir;
};

struct RayHit InterSphere(struct Ray* r, float3 spos, float srad){
    struct RayHit hit;
    hit.t = MAX_RENDER_DIST;
    float3 l = spos - r->pos;
    float tca = dot(r->dir, l);
    float d = tca*tca - dot(l, l) + srad*srad;
    if(d < 0) return hit;
    float t = tca - sqrt(d);
    if(t < 0){
        t = tca + sqrt(d);
        if(t < 0) return hit;
    }
    hit.t = t;
    hit.pos = r->pos + r->dir * t;
    hit.nor = (hit.pos - spos) / srad;
    return hit;
}

#ifdef GLINTEROP
__kernel void render(
    write_only image2d_t image_buffer,
#else
__kernel void render(
    __global int *image_buffer,
#endif
    __global const float *sc_spheres,
    const uint w,
    const uint h,
    const uint sc_spheres_count
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    uint pixid = x + y * w;
    //(0,0) is in middle of screen
    float2 uv = (float2)(((float)x / w) - 0.5f, ((float)y / h) - 0.5f);
    uv.x *= ((float)w/h);
    //colours
    char r = 0;
    char g = 0;
    char b = 0;
    //construct ray, simple perspective
    struct Ray ray;
    ray.pos = (float3)(0,0,0);
    ray.dir = normalize((float3)(uv.x,uv.y,-1) - ray.pos);
    //intersect all spheres
    float col = MAX_RENDER_DIST;
    for(int i = 0; i < sc_spheres_count; i++){
        int off = i * 4;
        float3 spos = (float3)(sc_spheres[off + 0], sc_spheres[off + 1], sc_spheres[off + 2]);
        float srad = sc_spheres[off + 3];
        struct RayHit hit = InterSphere(&ray, spos, srad);
        col = min(hit.t, col);
    }
    if(col >= MAX_RENDER_DIST) col = -1.0f;
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