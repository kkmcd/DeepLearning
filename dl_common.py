import numpy as np

def mean_squared_error(y,t):
    '''
    평균 제곱 오차를 반환합니다
    
     입력:
        y: 출력 데이터
        x: 타겟(레이블), 원 핫 인코딩
    '''
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error2(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
        
    batch_size = y.shape[0]
    
    if one_hot_encoding:
        return -np.sum(t*np.log(y+1e-7)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size),t])) / batch_size
    
   
