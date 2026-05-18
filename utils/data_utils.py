def visualize_heatmap_new(pre_hot_map, output_size=None):

    if output_size:
        fig_width_inch = output_size[0] / 100.0
        fig_height_inch = output_size[1] / 100.0
    else:
        fig_width_inch = pre_hot_map.shape[1] / 100.0
        fig_height_inch = pre_hot_map.shape[0] / 100.0

    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=100)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()

    cmap = 'jet'
    im = ax.imshow(pre_hot_map, cmap=cmap, interpolation='bicubic', aspect='auto', vmin=0, vmax=1)

    buf = io.BytesIO()

    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    img_pil = Image.open(buf)
    img_rgb = img_pil.convert('RGB')
    heatmap_img_np = np.array(img_rgb)

    if output_size is not None and (heatmap_img_np.shape[1], heatmap_img_np.shape[0]) != output_size:
        heatmap_img_np = cv2.resize(heatmap_img_np, output_size, interpolation=cv2.INTER_LINEAR)

    return heatmap_img_np