# data/

Veri dosyaları git'e **girmez** (bkz. `.gitignore`). Bu klasör yapı içindir.

Beklenen yerleşim:

```
data/
├── raw/                 # Roboflow export'ları (dokunulmaz)
├── uav_ldz/             # işlenmiş, eğitime hazır (YOLO formatı)
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
└── coco/                # COCO json'lar (eval için)
```

Export alma ve remap adımları: `notes.md` → Faz 1.
