// Debug Blend Factor Overlay Shader
// Reads the Post shader's predicted blend factor from r_debug_visualization (t19)
// and composites a heatmap overlay onto rw_mlsr_output_color (u2).
//
// Blend factor meaning (from sigmoid of model output filter[3]):
//   High (warm colors) = trusting history more (temporal reuse)
//   Low  (cool colors) = trusting current frame more (spatial reconstruction)
//
// If edges wobble, look for areas that flicker between warm and cool colors —
// those are pixels where the model is unstable in its temporal blend decision.

// We only need two resources from the Pre/Post descriptor layout:
Texture2D<float>    r_debug_visualization : register(t19);
RWTexture2D<float3> rw_mlsr_output_color  : register(u2);

// Heatmap: blue (0.0) → cyan → green (0.5) → yellow → red (1.0)
float3 Heatmap(float t)
{
    t = saturate(t);
    float3 c;
    if (t < 0.25)
    {
        float s = t * 4.0;
        c = float3(0, s, 1);            // blue → cyan
    }
    else if (t < 0.5)
    {
        float s = (t - 0.25) * 4.0;
        c = float3(0, 1, 1.0 - s);      // cyan → green
    }
    else if (t < 0.75)
    {
        float s = (t - 0.5) * 4.0;
        c = float3(s, 1, 0);            // green → yellow
    }
    else
    {
        float s = (t - 0.75) * 4.0;
        c = float3(1, 1.0 - s, 0);      // yellow → red
    }
    return c;
}

[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID)
{
    float blendFactor = r_debug_visualization[tid];
    float3 outputColor = rw_mlsr_output_color[tid];

    // Luminance (Bt.709) as desaturated background
    float lum = dot(outputColor, float3(0.2126, 0.7152, 0.0722));

    // Heatmap: blend factor 0→1 maps to blue→red
    // The Post shader stores: blending_factor = 1 - sigmoid(filter[3])
    // So high values = more current frame (less temporal reuse)
    float3 heat = Heatmap(blendFactor);

    // Composite: 60% heatmap, 40% luminance for spatial context
    float3 result = lerp(float3(lum, lum, lum), heat, 0.6);

    rw_mlsr_output_color[tid] = result;
}
