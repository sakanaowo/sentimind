import pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm
from multiprocessing import freeze_support
import warnings
from pathlib import Path

# Tắt các cảnh báo không cần thiết của thư viện transformers
warnings.filterwarnings("ignore", category=UserWarning)

def augment_minority_classes(df, text_col, label_col, aug_model, classes_to_augment):
    """
    Hàm sinh thêm dữ liệu cho các lớp thiểu số (ĐÃ TỐI ƯU CHẠY THEO BATCH 32 CÂU/LẦN)
    """
    augmented_data = []
    
    # Tạo từ điển map giữa label và label_id từ dữ liệu gốc
    label_id_map = {}
    if 'label_id' in df.columns:
        mapping_df = df[[label_col, 'label_id']].drop_duplicates().set_index(label_col)
        label_id_map = mapping_df['label_id'].to_dict()

    for target_label, num_samples in classes_to_augment.items():
        print(f"\n---> Đang xử lý lớp: '{target_label}' (Cần sinh thêm {num_samples} mẫu)")
        
        # Lọc dữ liệu của lớp mục tiêu
        target_df = df[df[label_col] == target_label]
        
        if len(target_df) == 0:
            print(f"Cảnh báo: Không tìm thấy mẫu cho lớp '{target_label}'. Bỏ qua.")
            continue

        # Lấy mẫu ngẫu nhiên
        samples_to_augment = target_df.sample(n=num_samples, replace=True)
        
        # CHUYỂN TOÀN BỘ TEXT THÀNH LIST ĐỂ GOM BATCH
        texts_list = samples_to_augment[text_col].tolist()
        batch_size = 32  # Gom 32 câu ném vào GPU 1 lần
        
        # THANH TIẾN TRÌNH NHẢY THEO BATCH (Ví dụ: 3000 câu chia 32 = ~94 bước)
        for i in tqdm(range(0, len(texts_list), batch_size), desc=f"Tiến độ ({target_label})"):
            # Lấy ra 32 câu
            batch_texts = texts_list[i : i + batch_size]
            try:
                # Đưa cả 32 câu vào dịch cùng lúc
                augmented_batch = aug_model.augment(batch_texts)
                
                # Lưu kết quả của cả 32 câu vào list tổng
                for new_text in augmented_batch:
                    new_row = {
                        text_col: new_text,
                        label_col: target_label
                    }
                    if 'label_id' in df.columns and target_label in label_id_map:
                        new_row['label_id'] = label_id_map[target_label]
                    augmented_data.append(new_row)
                    
            except Exception as e:
                print(f"Bỏ qua batch {i} do lỗi: {e}")
                continue
                
    return pd.DataFrame(augmented_data)


if __name__ == '__main__':
    freeze_support()
    
    # ==========================================
    # 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ
    # ==========================================
    
    current_dir = Path(__file__).parent
    
    # Thư mục chứa 3 file train, val, test gốc
    input_dir = current_dir.parent / "data" / "processed" 
    
    # File xuất ra
    output_file = current_dir.parent / "data" / "processed" / "Augmented_Combined_Data.csv"
    output_file = str(output_file)
    
    files_to_merge = ["train.csv", "val.csv", "test.csv"]
    
    # Cấu hình số lượng mẫu cần sinh thêm
    augmentation_config = {
        'Personality disorder': 3500,  
        'Stress': 6000,                
        'Bipolar': 6000,           
        'Anxiety': 5000                
    }

    # ==========================================
    # 2. ĐỌC VÀ GỘP DỮ LIỆU GỐC
    # ==========================================
    print("1. Đang tìm và gộp dữ liệu gốc...")
    dataframes = []
    
    for file_name in files_to_merge:
        file_path = input_dir / file_name
        
        if file_path.exists():
            df_temp = pd.read_csv(str(file_path))
            dataframes.append(df_temp)
            print(f"Đã đọc '{file_name}': {len(df_temp)} dòng.")
        else:
            print(f"Không tìm thấy file '{file_name}' tại {input_dir}")
            
    if not dataframes:
        print("\nLỖI NGHIÊM TRỌNG: Không đọc được file nào. Vui lòng kiểm tra lại.")
        exit()
        
    df = pd.concat(dataframes, ignore_index=True)
    print(f"\nĐã gộp thành công. Tổng số dòng dữ liệu: {len(df)}")
    print("\nPhân bố nhãn hiện tại:\n", df['label'].value_counts())

    # ==========================================
    # 3. KHỞI TẠO MÔ HÌNH DỊCH THUẬT
    # ==========================================
    print("\n2. Đang khởi tạo mô hình Back-Translation (English -> French -> English)...")
    
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-fr', 
        to_model_name='Helsinki-NLP/opus-mt-fr-en',   
        device='cuda',
        batch_size=32,      # Tận dụng tối đa sức mạnh GPU
        max_length=128      # Ép dịch nhanh, hoàn hảo cho BERTweet
    )

    # ==========================================
    # 4. CHẠY AUGMENTATION
    # ==========================================
    print("\n3. Bắt đầu quá trình Augmentation (Có thể mất một lúc)...")
    df_augmented = augment_minority_classes(
        df=df, 
        text_col='text', 
        label_col='label', 
        aug_model=back_translation_aug, 
        classes_to_augment=augmentation_config
    )

    # ==========================================
    # 5. GỘP, TRỘN (SHUFFLE) VÀ LƯU KẾT QUẢ
    # ==========================================
    if not df_augmented.empty:
        print("\n4. Đang gộp và xử lý dữ liệu cuối...")
        
        df_final = pd.concat([df, df_augmented], ignore_index=True)
        
        # Đảm bảo cột label_id được đưa về kiểu số nguyên (int) vì lúc concat có thể bị đổi thành float
        if 'label_id' in df_final.columns:
            df_final['label_id'] = df_final['label_id'].astype(int)
            
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        

        # Lưu ra file mới
        df_final.to_csv(output_file, index=False, quoting=1) # quoting=1 để đảm bảo các câu có dấu phẩy được bọc trong ngoặc kép
        
        print(f"\nHOÀN THÀNH! Dữ liệu mới đã được lưu vào: {output_file}")
        print(f"Tổng số mẫu hiện tại: {len(df_final)}")
        print("\nPhân bố nhãn MỚI:\n", df_final['label'].value_counts())
    else:
        print("\nKhông có dữ liệu mới nào được sinh ra.")