from streamlit_drawable_canvas import CanvasResult


def get_object_from_st_canvas(canvas_result: CanvasResult):
    if canvas_result.json_data is None:
        return ([], [], [])

    rects = list(
        filter(
            lambda obj: obj["type"] == "rect",
            canvas_result.json_data.get("objects", []),
        )
    )

    true_fgs = list(
        filter(
            lambda obj: obj["type"] == "path" and obj["stroke"] == "rgb(0, 255, 0)",
            canvas_result.json_data.get("objects", []),
        )
    )

    true_bgs = list(
        filter(
            lambda obj: obj["type"] == "path" and obj["stroke"] == "rgb(255, 0, 0)",
            canvas_result.json_data.get("objects", []),
        )
    )

    return (rects, true_fgs, true_bgs)