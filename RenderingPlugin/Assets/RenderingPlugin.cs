using UnityEngine;
using System.Collections;
using System.Runtime.InteropServices;
using System.IO;
using System;

public class RenderingPlugin : MonoBehaviour
{
    public static Mesh mesh;
    private static float FREQ = 4.0f;
    private static int MESH_SIZE = 11;
    private static int UNITY_RENDER_EVENT_ID = 0;
    private static String NORMAL_TEXTURE_ID = "_BumpMap";
    private static Texture2D tex, nTex;
    public bool useMeshURL = false;
    private Vector2 oldMousePos;
    private bool isRotating = false;
    private static float ROTATION_SCALE = 5;

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void MyDelegate(string str);

    [DllImport("RenderingPluginDLL")]
    private static extern void UnityRenderEvent(int eventID);
    [DllImport("RenderingPluginDLL")]
    private static extern void Init(int sideSize, IntPtr texPtr, IntPtr nTexPtr, [In, Out] int[] triangles, int trCount);
    [DllImport("RenderingPluginDLL")]
    private static extern void Cleanup();
    [DllImport("RenderingPluginDLL")]
    private static extern void SetDebugFunction(IntPtr fp);
    [DllImport("RenderingPluginDLL")]
    private static extern void ComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void ParallelComputeSineWave([In, Out] Vector3[] verts, float time);
    [DllImport("RenderingPluginDLL")]
    private static extern void SetTimeFromUnity(float t);

    public string meshURL;
    Vector3[] verts;
    int[] triangles;

    int numVerts;
    int numFaces;

    IEnumerator Start()
    {
        if (useMeshURL)
        {
            if (meshURL == null)
            {
                Debug.Log("Please choose a mesh to render!");
                yield return null;
            }
            else getMeshDataFromFile(meshURL);
        }
        else
        {
            MESH_SIZE = (int)Math.Sqrt(GetComponent<MeshFilter>().mesh.vertexCount);

            mesh = GetComponent<MeshFilter>().mesh;
        }

        verts = GetComponent<MeshFilter>().mesh.vertices;

        MyDelegate callbackDelegate = new MyDelegate(callback);
        IntPtr intptrDelegate = Marshal.GetFunctionPointerForDelegate(callbackDelegate);
        SetDebugFunction(intptrDelegate);

        byte[] textureData = getBytesFromVector3Array(verts);
        tex = new Texture2D(MESH_SIZE, MESH_SIZE, TextureFormat.RGBAFloat, false);
        FillTextureWithData(tex, verts, false);
        tex.Apply();
        GetComponent<Renderer>().material.mainTexture = tex;

        triangles = mesh.triangles;
        nTex = new Texture2D(MESH_SIZE, MESH_SIZE, TextureFormat.RGBAFloat, false);
        FillTextureWithData(nTex, mesh.normals, true);
        nTex.Apply();
        GetComponent<Renderer>().material.SetTexture(NORMAL_TEXTURE_ID, nTex);

        Init(MESH_SIZE, tex.GetNativeTexturePtr(), nTex.GetNativeTexturePtr(), mesh.triangles, mesh.triangles.Length);

        yield return StartCoroutine("CallPluginAtEndOfFrames");
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(1))
        {
            isRotating = true;
            oldMousePos = Input.mousePosition;
        }
        else if (Input.GetMouseButtonUp(1)) isRotating = false;
        else if (isRotating) RotateMesh();
    }

    void RotateMesh()
    {
        Vector2 mousePos = new Vector2( Input.mousePosition.x, Input.mousePosition.y );
        Vector2 delta = mousePos - oldMousePos;
        oldMousePos = mousePos;

        transform.Rotate(-Vector3.right * delta.y * Time.deltaTime * ROTATION_SCALE);
        transform.Rotate(Vector3.forward * delta.x * Time.deltaTime * ROTATION_SCALE);
    }

    void OnApplicationQuit()
    {
        Cleanup();
    }

    void getMeshDataFromFile(string url)
    {
        mesh = new Mesh();

        getVertsFromFile(url);
        getFacesFromFile(url);

        mesh.vertices = verts;
        mesh.triangles = triangles;

        mesh.RecalculateNormals();

        GetComponent<MeshFilter>().mesh = mesh;
    }

    void getVertsFromFile(string url) 
    {
        numVerts = 0;

        string[] lines = File.ReadAllLines(url);
        foreach(string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "v")
            {
                numVerts++;
            }
        }

        int index = 0;

        verts = new Vector3[numVerts];

        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "v")
            {
                verts[index++] = new Vector3(float.Parse(tokens[1]), float.Parse(tokens[2]), float.Parse(tokens[3]));
            }
        }
    }

    void getFacesFromFile(string url)
    {
        numFaces = 0;

        string[] lines = File.ReadAllLines(url);
        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "f")
            {
                numFaces++;
            }
        }

        triangles = new int[numFaces * 3];

        int index = 0;

        foreach (string line in lines)
        {
            string[] tokens = line.Split(' ');
            if (tokens[0] == "f")
            {
                triangles[index++] = int.Parse(tokens[1]) - 1;
                triangles[index++] = int.Parse(tokens[2]) - 1;
                triangles[index++] = int.Parse(tokens[3]) - 1;
            }
        }
    }

    static void callback(string str)
    {
        Debug.Log("Callback: " + str);
    }

    byte[] getBytesFromVector3Array(Vector3[] array)
    {
        byte[] bytes = new byte[array.Length * 4 * 4];

        int index = 0;
        float one = 1.0f;

        for (int i = 0; i < array.Length; i++)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].x), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].y), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(array[i].z), 0, bytes, index, 4);
            index += 4;
            Buffer.BlockCopy(BitConverter.GetBytes(one), 0, bytes, index, 4);
            index += 4;
        }

        return bytes;
    }

    private void FillTextureWithData(Texture2D tex, Vector3[] vertices, bool n)
    {
        float alpha = n ? 0.0f : 1.0f;

        for(int i = 0; i < vertices.Length; i++){
            Vector3 v = vertices[i];
            Color color = new Color(v[0], v[1], v[2], alpha);
            tex.SetPixel(i / MESH_SIZE, i % MESH_SIZE, color);
        }
    }

    private IEnumerator CallPluginAtEndOfFrames()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();

            SetTimeFromUnity(Time.timeSinceLevelLoad);

            GL.IssuePluginEvent(UNITY_RENDER_EVENT_ID);
        }
    }
}