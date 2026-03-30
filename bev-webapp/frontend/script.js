function previewImage(inputId, previewId) {
    const file = document.getElementById(inputId).files[0];
    const preview = document.getElementById(previewId);

    if (file) {
        preview.src = URL.createObjectURL(file);
    } else {
        preview.src = "";
    }
}

// ✅ Wait for DOM to load
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("img1").addEventListener("change", () => previewImage("img1", "preview1"));
    document.getElementById("img2").addEventListener("change", () => previewImage("img2", "preview2"));
    document.getElementById("img3").addEventListener("change", () => previewImage("img3", "preview3"));
});


async function uploadImages() {

    const img1 = document.getElementById("img1").files[0];
    const img2 = document.getElementById("img2").files[0];
    const img3 = document.getElementById("img3").files[0];
    const destination = document.getElementById("destination").value;

    if (!img1 || !img2 || !img3) {
        alert("Please upload all 3 images!");
        return;
    }

    const formData = new FormData();
    formData.append("img1", img1);
    formData.append("img2", img2);
    formData.append("img3", img3);
    formData.append("destination", destination);

    // Show loader
    document.getElementById("loader").style.display = "block";

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        console.log("Response:", data);

        // ❌ Backend error handling
        if (data.error) {
            alert("Backend Error: " + data.error);
            return;
        }

        const timestamp = new Date().getTime();

        // ✅ Update outputs (cache busting)
        document.getElementById("bevImg").src = data.bev + "?t=" + timestamp;
        document.getElementById("riskImg").src = data.risk + "?t=" + timestamp;
        document.getElementById("trajImg").src = data.trajectory + "?t=" + timestamp;

        // 🔥 NEW: Confidence Map
        if (data.confidence) {
            document.getElementById("confImg").src = data.confidence + "?t=" + timestamp;
        }

    } catch (error) {
        console.error("Error:", error);
        alert("Error running model. Check backend terminal.");
    }

    // Hide loader
    document.getElementById("loader").style.display = "none";
}