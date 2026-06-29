3 MODEL ĐỀ XUẤT CHO DỰ ÁN

Mục tiêu của thư mục này là mô tả 3 hướng model hợp lý nhất cho dự án "Latent Manipulation of 2D Brain MRI using Volume-Preserving GANs". Đây là đề xuất kiến trúc và chiến lược triển khai, không phải script chạy sẵn.

1. Age-Conditional StyleGAN2-ADA

Đây là model nên ưu tiên làm hướng chính.

Ý tưởng:

- Huấn luyện StyleGAN2-ADA trên các lát cắt MRI não 2D sau tiền xử lý bảo toàn thể tích.
- Đưa tuổi vào model dưới dạng điều kiện liên tục, ví dụ age normalized hoặc age embedding.
- Có thể thêm điều kiện slice index, giới tính hoặc site/scanner để giảm nhiễu do lát cắt và nguồn dữ liệu.
- Sau khi huấn luyện, thao tác latent vector trong W/W+ space để mô phỏng cùng một cấu trúc não ở các tuổi khác nhau.

Vì sao hợp lý:

- StyleGAN2-ADA sinh ảnh 2D chất lượng cao và ổn định hơn GAN cơ bản.
- ADA hữu ích khi số lượng subject chỉ ở mức vài nghìn, như OpenBHB.
- Latent space của StyleGAN phù hợp với mục tiêu feature learning và latent manipulation.
- Dễ mở rộng bằng latent encoder như pSp/e4e để đưa ảnh MRI thật vào latent space rồi chỉnh tuổi.

Input đề xuất:

- Ảnh MRI 2D đã skull-strip/alignment/normalization.
- Tuổi chuẩn hóa về khoảng [0, 1] hoặc z-score.
- Slice index hoặc vị trí lát cắt.
- Giới tính/site/scanner nếu muốn kiểm soát confound.

Loss/đánh giá nên có:

- Adversarial loss, R1 regularization, path length regularization.
- Age consistency loss: ảnh sinh ra khi gán tuổi mục tiêu phải được age regressor độc lập dự đoán gần tuổi mục tiêu.
- Identity/structure preservation loss nếu dùng encoder để chỉnh ảnh thật.
- Kiểm tra volume proxy hoặc segmentation consistency trên chuỗi lát cắt.

Nguồn tham khảo:

- https://github.com/NVlabs/stylegan2-ada-pytorch

2. Age-Conditioned StarGAN v2 / MRI Aging Translator

Đây là model hợp lý nếu muốn nhấn mạnh mục tiêu "same anatomy, different ages".

Ý tưởng:

- Xem bài toán như image-to-image translation: đầu vào là lát cắt MRI của một người, đầu ra là cùng lát cắt đó ở một nhóm tuổi mục tiêu.
- Chia tuổi thành các age domain, ví dụ: trẻ, thanh niên, trung niên, cao tuổi. Có thể dùng age bin thay vì tuổi liên tục để huấn luyện dễ hơn.
- Dùng StarGAN v2 hoặc biến thể conditional CycleGAN để học chuyển đổi đa miền tuổi bằng một model duy nhất.

Vì sao hợp lý:

- Trực tiếp phục vụ thao tác tuổi trên ảnh MRI đầu vào.
- Có thể dùng identity loss, cycle loss và structural loss để giữ giải phẫu cá nhân.
- Phù hợp để trình bày kết quả trực quan: cùng một anatomy, nhiều tuổi mục tiêu.

Input đề xuất:

- Ảnh MRI 2D của subject gốc.
- Age domain nguồn và age domain đích.
- Mask não để tính structural loss trong vùng não.

Loss/đánh giá nên có:

- Adversarial loss theo domain tuổi.
- Style reconstruction hoặc latent reconstruction loss.
- Cycle consistency hoặc identity loss.
- SSIM/L1 trong brain mask để giữ cấu trúc.
- Age consistency loss bằng age regressor độc lập.
- Volume/segmentation consistency để tránh model chỉ phóng to/thu nhỏ não.

Nguồn tham khảo:

- https://github.com/clovaai/stargan-v2
- https://arxiv.org/abs/1912.01865

3. Conditional Latent Diffusion / Medfusion-style Baseline

Đây là model nên dùng làm baseline mạnh hoặc hướng mở rộng nếu GAN sinh ảnh mờ/khó hội tụ.

Ý tưởng:

- Huấn luyện autoencoder để nén lát cắt MRI vào latent space.
- Huấn luyện diffusion model trong latent space, có điều kiện theo tuổi, slice index và các biến kiểm soát.
- Khi sinh ảnh, điều khiển tuổi bằng conditioning thay vì chỉnh trực tiếp latent vector như GAN.

Vì sao hợp lý:

- Diffusion thường ổn định hơn GAN và ít bị mode collapse.
- Latent diffusion giảm chi phí so với diffusion trực tiếp trên ảnh độ phân giải cao.
- Có thể tạo ảnh đa dạng và làm baseline so sánh với GAN.
- Phù hợp với dữ liệu y sinh nếu cần ưu tiên tính đa dạng và ổn định huấn luyện.

Hạn chế:

- Không bám sát chữ "GAN" trong theme bằng hai model trên.
- Latent manipulation không trực quan bằng StyleGAN W/W+ space.
- Tốc độ sampling thường chậm hơn GAN.

Input đề xuất:

- Ảnh MRI 2D đã tiền xử lý.
- Tuổi liên tục hoặc age embedding.
- Slice index, giới tính/site/scanner nếu cần.

Loss/đánh giá nên có:

- Reconstruction loss cho autoencoder.
- Diffusion noise prediction loss.
- Age consistency bằng age regressor độc lập.
- FID/KID, SSIM nếu có reconstruction, và kiểm tra segmentation/volume.

Nguồn tham khảo:

- https://arxiv.org/abs/2112.10752
- https://www.nature.com/articles/s41598-023-39278-0

Khuyến nghị triển khai

Nếu chỉ chọn một model để làm chính, nên chọn Age-Conditional StyleGAN2-ADA vì sát nhất với theme_GenAI.pdf: GAN, latent space, thao tác tuổi và sinh ảnh MRI 2D. Nếu cần một model thể hiện rõ "same anatomy, different ages", thêm StarGAN v2/MRI Aging Translator. Nếu cần baseline hiện đại để so sánh chất lượng sinh ảnh, thêm Conditional Latent Diffusion.
