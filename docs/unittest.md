![corverage](./uniiiiiiiiiiiitest.jpg.jpg)

Các test case đã viết:

    Test hàm trích xuất đặc trưng (so sánh với file expect JSON).

        Thách thức: sai số số học → giải pháp: làm tròn đến 6 chữ số thập phân.

    Test hàm predi() (so sánh DataFrame với expect CSV).

        Thách thức: dtype phức tạp → giải pháp: dùng pd.testing.assert_frame_equal() bỏ qua dtype.
