using UnityEditor;
using UnityEngine;


class RenderingPluginOptions
{
    static RenderingPlugin rp = GameObject.Find("PluginGameObject").GetComponent<RenderingPlugin>();

    [MenuItem("Rendering Plugin/Mesh")]
    private static void ApplyMesh () 
    {
        string path = EditorUtility.OpenFilePanel(
					"Choose mesh to render",
					"/Assets/Meshes",
					"obj");
        
        if (path != null && path != "") rp.SetMeshURL(path);
	}

    [MenuItem("Rendering Plugin/Shader/GLSL Shader")]
    private static void ApplyGLSLShader ()
    {
        rp.SetShader("Shaders/PluginShader");
    }

    [MenuItem("Rendering Plugin/Shader/Surface Shader")]
    private static void ApplySurfaceShader ()
    {
        rp.SetShader("Shaders/PluginShader2");
    }
}
