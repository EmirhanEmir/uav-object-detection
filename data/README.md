# data/

Veri dosyaları git'e **girmez** (bkz. `.gitignore`). Bu klasör yapı içindir.

Beklenen yerleşim:

```
data/
├── raw/                 # etiketli export'lar (dokunulmaz)
├── uav_ldz/             # işlenmiş, eğitime hazır (YOLO formatı)
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
└── coco/                # COCO json'lar (eval için)
```

Export alma, sınıf remap ve dönüştürme adımları yerelde yürütülür; bu adımların
kodu ve verinin kendisi depoya dahil değildir.
