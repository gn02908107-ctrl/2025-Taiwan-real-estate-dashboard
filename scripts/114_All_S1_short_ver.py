import pandas as pd
import os

#------坪數換算------
def add_pin_columns(df):
    df_copy = df.copy()
    df_copy['建物移轉總面積平方公尺'] = pd.to_numeric(df_copy['建物移轉總面積平方公尺'],errors = 'coerce')
    df_copy['單價元平方公尺'] = pd.to_numeric(df_copy['單價元平方公尺'],errors = 'coerce')
    df_copy['總坪數'] = (df_copy['建物移轉總面積平方公尺']*0.3025).round(2)
    df_copy['單價_萬元每坪'] = (df_copy['單價元平方公尺']*3.30578/10000).round(2)
    return df_copy

#------讀取檔案並初步篩選------
def load_and_select(file_path):
    df = pd.read_csv(file_path,on_bad_lines='warn')
    cols = ['鄉鎮市區','交易標的','土地位置建物門牌','主要用途','建物移轉總面積平方公尺','總價元','單價元平方公尺']
    return df[cols].iloc[1:]

#------依交易標的及主要用途篩選------
def filter_by_type(df,deal_type):
    target = '房地(土地+建物)+車位' if deal_type == 'with_car' else '房地(土地+建物)'
    result = df[(df['交易標的'] == target) & (df['主要用途'] == '住家用')].copy()
    return result

#------依行政區排序------
def sort_by_district(df):
    return df.sort_values(by = '鄉鎮市區')

#------輸出檔案------
def save_csv(df,output_path):
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'資料已儲存至:{output_path}')
    
#------整合完整處理流程------
def process_file(input_path,label):
    
    #檔案不存在就跳過
    if not os.path.exists(input_path):
        print(f'找不到檔案並跳過:{input_path}')
        return
    
    #讀取與初步篩選
    datas = load_and_select(input_path)
    print(datas)
    print('-'*70)
    
    #坪數與單價換算
    datas = add_pin_columns(datas)
    
    #含車位資料
    with_car = filter_by_type(datas, 'with_car')
    print(with_car)
    print('*'*70)
    
    #不含車位資料
    no_car = filter_by_type(datas, 'no_car')
    print(no_car)
    print('*'*70)
    
    #排序
    with_car = sort_by_district(with_car)
    no_car = sort_by_district(no_car)
    
    #輸出資料
    save_csv(with_car,f'S4_澎湖縣房地產交易資料_含車位({label}).csv')
    save_csv(no_car,f'S4_澎湖縣房地產交易資料_不含車位({label}).csv')
    
#------主程式------
def main():
    process_file('./season 4/x_lvr_land_a.csv', '中古屋')
    process_file('./season 4/x_lvr_land_b.csv', '預售屋')
    
if __name__ == '__main__':
    main()