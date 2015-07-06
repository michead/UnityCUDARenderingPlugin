using UnityEditor;
using UnityEngine;


class RenderingPluginOptions
{
    [MenuItem("Rendering Plugin/Mesh")]
    private static void Apply () 
    {
        string path = EditorUtility.OpenFilePanel(
					"Choose mesh to render",
					"/Assets/Meshes",
					"obj");
        Debug.Log("Path:" + path);

        GameObject pluginObj = GameObject.Find("PluginGameObject");
        RenderingPlugin rp = pluginObj.GetComponent<RenderingPlugin>();
        if (path != null && path != "") rp.meshURL = path;
	}
}
