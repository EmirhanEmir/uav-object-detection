# data/

Veri dosyaları git'e **girmez** (bkz. `.gitignore`). Bu klasör yapı içindir.

Beklenen yerleşim:

```
data/
├── uav_ldz/             # eğitime hazır veri seti (YOLO formatı)
│   ├── train/{images,labels}
│   ├── val/{images,labels}
│   └── test/{images,labels}
├── holdout/images/      # eğitimde hiç görülmeyen ayrı kaynak — ana değerlendirme seti
└── coco/                # COCO GT json'ları (instances_{train,val,test,holdout}.json)
```

Veri hazırlama adımları (export, sınıf eşleme, dönüştürme) ve verinin kendisi depoya
dahil değildir. Depo tespit ve değerlendirme koduna odaklanır.
