using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// XR Bolt Inspection Client for Unity (Meta Quest / HoloLens / AR Foundation)
/// Attaches to XR camera or inspection controller to send real-time sensor
/// data to the Python FastAPI backend and update 3D AR holograms.
/// </summary>
public class XRBoltInspectionClient : MonoBehaviour
{
    [Header("Backend API Configuration")]
    [SerializeField] private string apiBaseUrl = "http://localhost:8000";

    [Header("AR Indicator Settings")]
    [SerializeField] private GameObject greenTightPrefab;
    [SerializeField] private GameObject yellowVerifyPrefab;
    [SerializeField] private GameObject redDefectPrefab;

    [System.Serializable]
    public class VisualInspectPayload
    {
        public string equipment_id;
        public string component_id;
        public float[] position_3d;
        public float gap_mm;
        public float contact_score;
        public float vision_quality;
        public float depth_quality;
        public float registration_quality;
    }

    [System.Serializable]
    public class InspectionResponse
    {
        public string equipment_id;
        public string component_id;
        public string initial_visual_prediction;
        public float loose_probability;
        public float confidence;
        public string decision;
        public string final_result;
        public string verification_mode;
    }

    [System.Serializable]
    public class SpatialDefectItem
    {
        public string equipment_id;
        public string component_id;
        public Vector3Position position;
        public string defect_type;
        public float probability;
        public float confidence;
        public string verification;
        public string timestamp;
    }

    [System.Serializable]
    public class Vector3Position
    {
        public float x;
        public float y;
        public float z;
    }

    /// <summary>
    /// Trigger an optical XR scan on a targeted bolt
    /// </summary>
    public void ScanBolt(string boltId, Vector3 worldPosition, float measuredGapMm, float contactScore = 0.95f)
    {
        StartCoroutine(SendVisualScanRoutine(boltId, worldPosition, measuredGapMm, contactScore));
    }

    private IEnumerator SendVisualScanRoutine(string boltId, Vector3 pos, float gap, float contact)
    {
        string endpoint = $"{apiBaseUrl}/api/inspect/visual";

        VisualInspectPayload payload = new VisualInspectPayload
        {
            equipment_id = "MACHINE_01",
            component_id = boltId,
            position_3d = new float[] { pos.x, pos.y, pos.z },
            gap_mm = gap,
            contact_score = contact,
            vision_quality = 0.92f,
            depth_quality = 0.88f,
            registration_quality = 0.95f
        };

        string jsonBody = JsonUtility.ToJson(payload);

        using (UnityWebRequest request = new UnityWebRequest(endpoint, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonBody);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.Success)
            {
                InspectionResponse res = JsonUtility.FromJson<InspectionResponse>(request.downloadHandler.text);
                Debug.Log($"[XR Inspection] Bolt {boltId} -> Result: {res.final_result}, Decision: {res.decision}");
                ApplyHolographicIndicator(boltId, pos, res);
            }
            else
            {
                Debug.LogError($"[XR Inspection] Request failed: {request.error}");
            }
        }
    }

    private void ApplyHolographicIndicator(string boltId, Vector3 pos, InspectionResponse res)
    {
        // Choose holographic marker based on decision
        GameObject markerPrefab = greenTightPrefab;
        if (res.decision == "PHYSICAL_VERIFICATION")
        {
            markerPrefab = yellowVerifyPrefab;
        }
        else if (res.decision == "LIKELY_DEFECT" || res.decision == "CONFIRMED_DEFECT" || res.final_result.Contains("LOOSE"))
        {
            markerPrefab = redDefectPrefab;
        }

        if (markerPrefab != null)
        {
            GameObject pin = Instantiate(markerPrefab, pos + Vector3.up * 0.1f, Quaternion.identity);
            pin.name = $"AR_Pin_{boltId}_{res.final_result}";
        }
    }
}
