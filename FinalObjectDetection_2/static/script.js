document.addEventListener("DOMContentLoaded", () => {
    const colormapSelect = document.getElementById("colormap");
    const videoFeed = document.getElementById("videoFeed");
    const temperatureDiv = document.getElementById("temperature");
    const tempValueSpan = document.getElementById("temp-value");
    const heatmapLegendImg = document.getElementById("heatmapLegendImg");
    const filterCheckbox = document.getElementById("filterCheckbox");

    function updateHeatmapLegend(colormap) {
        heatmapLegendImg.src = `/static/${colormap}_legend.png?${new Date().getTime()}`;
    }

    colormapSelect.addEventListener("change", () => {
        const colormap = colormapSelect.value;
        fetch("/change_colormap", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ colormap })
        }).then(() => {
            updateHeatmapLegend(colormap);
        });
    });

    let debounceTimeout;
    videoFeed.addEventListener("mousemove", (event) => {
        if (debounceTimeout) {
            clearTimeout(debounceTimeout);
        }
        debounceTimeout = setTimeout(() => {
            const rect = videoFeed.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            fetch("/temperature", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: `x=${x}&y=${y}`,
            })
            .then(response => response.json())
            .then(data => {
                tempValueSpan.textContent = data.temperature.toFixed(2);
                temperatureDiv.style.display = 'block';
                temperatureDiv.style.left = `${event.clientX + 10}px`;
                temperatureDiv.style.top = `${event.clientY - 30}px`;
            });
        }, 50);
    });

    videoFeed.addEventListener("mouseleave", () => {
        temperatureDiv.style.display = 'none';
    });

    filterCheckbox.addEventListener("change", () => {
        fetch("/filter", {
            method: "GET"
        });
    });

    updateHeatmapLegend(colormapSelect.value);
});


